#!/usr/bin/env python3
"""Generate the confined-cylinder DNS/particle-pair database used by gDMD.

The script solves the two-dimensional confined cylinder problem, advects passive
tracers, and stores paired particle and velocity snapshots.  It intentionally
does not perform DMD or gap-field post-processing; see
``scripts/run_cylinder2d_gdmd.py`` for that step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path




@dataclass
class CylinderDNSConfig:
    """Run parameters for the confined-cylinder particle-tracking simulation."""

    final_time: float = 300.0
    dt: float = 1.0 / 1600.0
    save_freq: int = 50
    steady_fraction: float = 0.01
    resolution_divisor: int = 3
    mu: float = 0.001
    rho: float = 1.0
    inlet_velocity: float = 1.0
    random_dist: bool = True
    stdHdiv: int = 3
    newseed_rate: int = 1
    newseed_num: int = 8
    seed: int = 1
    output_dir: Path = Path("data")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-time", type=float, default=CylinderDNSConfig.final_time, help="Final simulation time.")
    parser.add_argument("--dt", type=float, default=CylinderDNSConfig.dt, help="Time step.")
    parser.add_argument("--save-freq", type=int, default=CylinderDNSConfig.save_freq, help="Store one image pair every SAVE_FREQ time steps.")
    parser.add_argument("--steady-fraction", type=float, default=CylinderDNSConfig.steady_fraction, help="Fraction of final time discarded as transient.")
    parser.add_argument("--resolution-divisor", type=int, default=CylinderDNSConfig.resolution_divisor, help="Near-cylinder element size R/divisor.")
    parser.add_argument("--mu", type=float, default=CylinderDNSConfig.mu, help="Dynamic viscosity.")
    parser.add_argument("--rho", type=float, default=CylinderDNSConfig.rho, help="Density.")
    parser.add_argument("--inlet-velocity", type=float, default=CylinderDNSConfig.inlet_velocity, help="Uniform inlet/wall velocity.")
    parser.add_argument("--random-dist", action=argparse.BooleanOptionalAction, default=CylinderDNSConfig.random_dist, help="Use a Gaussian y-distribution for passive-particle seeding.")
    parser.add_argument("--uniform-seeds", action="store_false", dest="random_dist", help="Compatibility alias for --no-random-dist.")
    parser.add_argument("--stdHdiv", type=int, default=CylinderDNSConfig.stdHdiv, help="Seed-distribution standard deviation denominator: sigma=H/stdHdiv.")
    parser.add_argument("--newseed-num", type=int, default=CylinderDNSConfig.newseed_num, help="Number of particles reseeded at each reseeding step.")
    parser.add_argument("--newseed-rate", type=int, default=CylinderDNSConfig.newseed_rate, help="Particle reseeding interval in time steps.")
    parser.add_argument("--seed", type=int, default=CylinderDNSConfig.seed, help="Random seed for passive-particle initialization/reseeding.")
    parser.add_argument("--output-dir", type=Path, default=CylinderDNSConfig.output_dir, help="Directory where the .npz output is written.")
    return parser


def config_from_args(args: argparse.Namespace) -> CylinderDNSConfig:
    return CylinderDNSConfig(
        final_time=args.final_time,
        dt=args.dt,
        save_freq=args.save_freq,
        steady_fraction=args.steady_fraction,
        resolution_divisor=args.resolution_divisor,
        mu=args.mu,
        rho=args.rho,
        inlet_velocity=args.inlet_velocity,
        random_dist=args.random_dist,
        stdHdiv=args.stdHdiv,
        newseed_num=args.newseed_num,
        newseed_rate=args.newseed_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def run_simulation(cfg: CylinderDNSConfig) -> None:
    """Run the DNS and write the paired snapshot database."""

    import gmsh
    import numpy as np
    import tqdm.autonotebook
    from basix.ufl import element
    from dolfinx.fem import (
        Constant,
        Function,
        assemble_scalar,
        dirichletbc,
        form,
        functionspace,
        locate_dofs_topological,
    )
    from dolfinx.fem.petsc import (
        apply_lifting,
        assemble_matrix,
        assemble_vector,
        create_matrix,
        create_vector,
        set_bc,
    )
    from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from petsc4py import PETSc
    from ufl import (
        FacetNormal,
        Measure,
        TestFunction,
        TrialFunction,
        as_vector,
        div,
        dot,
        dx,
        grad,
        inner,
        lhs,
        nabla_grad,
        rhs,
    )

    gmsh.initialize()
    try:
        # ============================================================
        # Geometry: confined cylinder
        # ============================================================
        L = 2.2
        H = 0.41
        xmin, xmax = 0.0, L
        ymin, ymax = 0.0, H
        gdim = 2

        # Cylinder radius / diameter
        R = 0.05
        r = R
        D = 2.0 * R

        # Cylinder center
        c_x = 0.2
        c_y = 0.2
        centers = np.array([[c_x, c_y]], dtype=float)

        # Mesh controls
        resmindiv = cfg.resolution_divisor
        res_min = r / resmindiv

        # ============================================================
        # Problem specifications
        # ============================================================
        t = 0.0
        T = cfg.final_time
        dt = cfg.dt
        Tsteady = cfg.steady_fraction * T
        save_freq = cfg.save_freq
        num_steps = int(T / dt)
        num_stored_alloc = int(num_steps / save_freq) + 1

        mu_value = cfg.mu
        rho_value = cfg.rho
        U_value = cfg.inlet_velocity

        # Seed parameters
        random_dist = cfg.random_dist
        stdHdiv = cfg.stdHdiv
        std_seed = H / stdHdiv
        newseed_rate = cfg.newseed_rate
        newseed_num = cfg.newseed_num
        newseed_bw = L / 100.0

        fname4npsave = (
            f"final_np_rmin{resmindiv:02d}_T{int(T):03d}"
            f"_stdH{stdHdiv:02d}_nsnum{newseed_num:02d}"
        )

        # ============================================================
        # Meshing
        # ============================================================
        mesh_comm = MPI.COMM_WORLD
        n_threads = mesh_comm.Get_size()
        model_rank = 0

        if mesh_comm.rank == model_rank:
            print(
                f"DNS time integration: T={T}, dt={dt}, "
                f"num_steps={num_steps}, save_freq={save_freq}",
                flush=True,
            )

        if mesh_comm.rank == model_rank:
            rectangle = gmsh.model.occ.addRectangle(xmin, ymin, 0.0, L, H, tag=1)
            cyl = gmsh.model.occ.addDisk(c_x, c_y, 0.0, r, r)
            gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, cyl)])
            gmsh.model.occ.synchronize()

        fluid_marker = 1
        if mesh_comm.rank == model_rank:
            volumes = gmsh.model.getEntities(dim=gdim)
            assert len(volumes) == 1
            gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
            gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
        inflow, outflow, walls, obstacle = [], [], [], []

        if mesh_comm.rank == model_rank:
            boundaries = gmsh.model.getBoundary(volumes, oriented=False)
            for boundary in boundaries:
                com = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])

                if np.allclose(com, [xmin, H / 2.0, 0.0]):
                    inflow.append(boundary[1])
                elif np.allclose(com, [xmax, H / 2.0, 0.0]):
                    outflow.append(boundary[1])
                elif (
                    np.allclose(com, [L / 2.0, ymax, 0.0]) or
                    np.allclose(com, [L / 2.0, ymin, 0.0])
                ):
                    walls.append(boundary[1])
                else:
                    obstacle.append(boundary[1])

            gmsh.model.addPhysicalGroup(1, walls, wall_marker)
            gmsh.model.setPhysicalName(1, wall_marker, "Walls")

            gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
            gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")

            gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
            gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")

            gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
            gmsh.model.setPhysicalName(1, obstacle_marker, "Cylinder")

        if mesh_comm.rank == model_rank:
            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)

            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
            gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2.0 * H)

            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        if mesh_comm.rank == model_rank:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.model.mesh.generate(gdim)
            gmsh.model.mesh.setOrder(2)
            gmsh.model.mesh.optimize("Netgen")

        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        imap = mesh.geometry.index_map()
        local_range = mesh.comm.gather(imap.local_range, root=0)
        xlocal = mesh.geometry.x[:imap.size_local, :2]
        xlocalgat = mesh.comm.gather(xlocal, root=0)
        ft.name = "Facet markers"

        if mesh.comm.rank == 0:
            xglobal = np.zeros((imap.size_global, 2), dtype=xlocal.dtype)
            for ith in range(n_threads):
                lrange = np.arange(local_range[ith][0], local_range[ith][1])
                xglobal[lrange] = xlocalgat[ith]

        # ============================================================
        # FE spaces and constants
        # ============================================================
        k = Constant(mesh, PETSc.ScalarType(dt))
        mu = Constant(mesh, PETSc.ScalarType(mu_value))
        rho = Constant(mesh, PETSc.ScalarType(rho_value))

        v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
        s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
        V = functionspace(mesh, v_cg2)
        Q = functionspace(mesh, s_cg1)

        fdim = mesh.topology.dim - 1

        # ============================================================
        # Boundary conditions
        # ============================================================
        class InletVelocity:
            def __init__(self, t):
                self.t = t

            def __call__(self, x):
                values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
                values[0] = U_value
                return values


        def cart2pol(x, y, x_offset, y_offset):
            xnew = x - x_offset
            ynew = y - y_offset
            rho = np.sqrt(xnew**2 + ynew**2)
            phi = np.arctan2(ynew, xnew)
            return rho, phi


        def pol2cart(rho, phi, x_offset, y_offset):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            x += x_offset
            y += y_offset
            return x, y


        u_inlet = Function(V)
        inlet_velocity = InletVelocity(t)
        u_inlet.interpolate(inlet_velocity)

        # Inlet: u = (1,0)
        bcu_inflow = dirichletbc(
            u_inlet,
            locate_dofs_topological(V, fdim, ft.find(inlet_marker))
        )

        # Top and bottom: same as original code, u = (1,0)
        bcu_walls = dirichletbc(
            u_inlet,
            locate_dofs_topological(V, fdim, ft.find(wall_marker))
        )

        # Cylinder: no-slip
        u_nonslip = np.array((0.0, 0.0), dtype=PETSc.ScalarType)
        bcu_obstacle = dirichletbc(
            u_nonslip,
            locate_dofs_topological(V, fdim, ft.find(obstacle_marker)),
            V
        )

        bcu = [bcu_inflow, bcu_walls, bcu_obstacle]

        # Outlet: pressure reference
        bcp_outlet = dirichletbc(
            PETSc.ScalarType(0),
            locate_dofs_topological(Q, fdim, ft.find(outlet_marker)),
            Q
        )
        bcp = [bcp_outlet]

        # ============================================================
        # Variables and projection scheme
        # ============================================================
        u = TrialFunction(V)
        v = TestFunction(V)
        u_ = Function(V)
        u_.name = "u"
        u_s = Function(V)
        u_n = Function(V)
        u_n1 = Function(V)

        p = TrialFunction(Q)
        q = TestFunction(Q)
        p_ = Function(Q)
        p_.name = "p"
        phi = Function(Q)

        f = Constant(mesh, PETSc.ScalarType((0.0, 0.0)))

        F1 = rho / k * dot(u - u_n, v) * dx
        F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
        F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
        F1 += dot(f, v) * dx

        a1 = form(lhs(F1))
        L1 = form(rhs(F1))
        A1 = create_matrix(a1)
        b1 = create_vector(L1)

        a2 = form(dot(grad(p), grad(q)) * dx)
        L2 = form(-rho / k * dot(div(u_s), q) * dx)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(L2)

        a3 = form(rho * dot(u, v) * dx)
        L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)

        # ============================================================
        # Solvers
        # ============================================================
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        # ============================================================
        # Diagnostics
        # ============================================================
        n = -FacetNormal(mesh)
        dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
        u_t = inner(as_vector((n[1], -n[0])), u_)

        drag = form(2.0 / D * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
        lift = form(-2.0 / D * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)

        if mesh.comm.rank == 0:
            C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
            C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
            t_u = np.zeros(num_steps, dtype=np.float64)
            t_p = np.zeros(num_steps, dtype=np.float64)

        tree = bb_tree(mesh, mesh.geometry.dim)
        points = np.array([[0.15, 0.2, 0.0], [0.25, 0.2, 0.0]])
        cell_candidates = compute_collisions_points(tree, points)
        colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
        front_cells = colliding_cells.links(0)
        back_cells = colliding_cells.links(1)

        if mesh.comm.rank == 0:
            p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)

        # ============================================================
        # Initial passive seeds
        # ============================================================
        rng = np.random.default_rng(cfg.seed)

        nseed0 = int(np.round(L * H / res_min**2 * newseed_num))
        seeds = rng.random([nseed0, 3])
        seeds[:, 0] *= L
        seeds[:, 1] *= H
        seeds[:, 2] = 0.0

        if random_dist:
            seeds[:, 1] = rng.normal(c_y, std_seed, nseed0)
            inside_y = np.logical_and(seeds[:, 1] > ymin, seeds[:, 1] < ymax)
            seeds = seeds[inside_y, :]

        seeds = mesh.comm.bcast(seeds, root=0)

        mask_outside = ((seeds[:, 0] - c_x) ** 2 + (seeds[:, 1] - c_y) ** 2 > r ** 2)
        seeds = seeds[mask_outside, :]

        nseeds = len(seeds)
        flag_store = False

        if mesh.comm.rank == 0:
            seeds_to_store1 = np.zeros((nseeds, 2, num_stored_alloc), dtype=PETSc.ScalarType)
            seeds_to_store2 = np.zeros((nseeds, 2, num_stored_alloc), dtype=PETSc.ScalarType)
            u_to_store1 = np.zeros((imap.size_global, 2, num_stored_alloc), dtype=PETSc.ScalarType)
            u_to_store2 = np.zeros((imap.size_global, 2, num_stored_alloc), dtype=PETSc.ScalarType)

        # ============================================================
        # Time integration with seed advection
        # ============================================================
        cfg.output_dir.mkdir(exist_ok=True, parents=True)

        progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
        ist1, ist2 = 0, 0

        for i in range(num_steps):
            progress.update(1)
            t += dt

            inlet_velocity.t = t
            u_inlet.interpolate(inlet_velocity)

            # Step 1: tentative velocity
            A1.zeroEntries()
            assemble_matrix(A1, a1, bcs=bcu)
            A1.assemble()

            with b1.localForm() as loc:
                loc.set(0)
            assemble_vector(b1, L1)
            apply_lifting(b1, [a1], [bcu])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, bcu)

            solver1.solve(b1, u_s.vector)
            u_s.x.scatter_forward()

            # Step 2: pressure correction
            with b2.localForm() as loc:
                loc.set(0)
            assemble_vector(b2, L2)
            apply_lifting(b2, [a2], [bcp])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, bcp)

            solver2.solve(b2, phi.vector)
            phi.x.scatter_forward()

            p_.vector.axpy(1, phi.vector)
            p_.x.scatter_forward()

            # Step 3: velocity correction
            with b3.localForm() as loc:
                loc.set(0)
            assemble_vector(b3, L3)
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            solver3.solve(b3, u_.vector)
            u_.x.scatter_forward()

            # Shift time levels
            with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
                loc_n.copy(loc_n1)
                loc_.copy(loc_n)

            # Diagnostics
            drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
            lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)

            p_front = None
            if len(front_cells) > 0:
                p_front = p_.eval(points[0], front_cells[:1])
            p_front = mesh.comm.gather(p_front, root=0)

            p_back = None
            if len(back_cells) > 0:
                p_back = p_.eval(points[1], back_cells[:1])
            p_back = mesh.comm.gather(p_back, root=0)

            if mesh.comm.rank == 0:
                t_u[i] = t
                t_p[i] = t - dt / 2.0
                C_D[i] = sum(drag_coeff)
                C_L[i] = sum(lift_coeff)

                for pressure in p_front:
                    if pressure is not None:
                        p_diff[i] = pressure[0]
                        break
                for pressure in p_back:
                    if pressure is not None:
                        p_diff[i] -= pressure[0]
                        break

            # Gather the velocity field only when it will be stored.
            # This avoids a costly MPI gather at every DNS time step.
            store_first = np.remainder(i, save_freq) == 0 and t > Tsteady
            store_second = np.remainder(i, save_freq) == 1 and flag_store
            ulocal = None
            if store_first or store_second:
                ulocal_local = u_.x.array[:int(2 * imap.size_local)].reshape(imap.size_local, 2)
                ulocal = mesh_comm.gather(ulocal_local, root=0)

            # --------------------------------------------------------
            # Seed advection
            # --------------------------------------------------------
            seeds_local = None
            seed_select = None

            cell_candidates_seeds = compute_collisions_points(tree, seeds)
            colliding_cells_seeds = compute_colliding_cells(mesh, cell_candidates_seeds, seeds)

            if len(colliding_cells_seeds.array) > 0:
                seed_select = np.zeros(len(colliding_cells_seeds.array), dtype=np.int32)
                cell_select = np.zeros(len(colliding_cells_seeds.array), dtype=np.int32)
                ctr = 0

                for ss in range(len(colliding_cells_seeds)):
                    if len(colliding_cells_seeds.links(ss)) > 0:
                        seed_select[ctr] = ss
                        cell_select[ctr] = colliding_cells_seeds.links(ss)[0]
                        ctr += 1

                seed_select = seed_select[:ctr]
                cell_select = cell_select[:ctr]
                seeds_local = seeds[seed_select].copy()

                u_dummy = u_.eval(seeds_local, cell_select)
                if u_dummy.ndim == 1:
                    seeds_local[:, :2] += u_dummy * dt
                else:
                    seeds_local[:, :2] += u_dummy * dt

            seeds_local = mesh.comm.gather(seeds_local, root=0)
            seed_select = mesh.comm.gather(seed_select, root=0)

            if mesh.comm.rank == 0:
                for ith in range(n_threads):
                    if seed_select[ith] is not None and seeds_local[ith] is not None:
                        seeds[seed_select[ith]] = seeds_local[ith]

            seeds = mesh.comm.bcast(seeds, root=0)

            # Reflect seeds outside the cylinder
            r_seeds, phi_seeds = cart2pol(seeds[:, 0], seeds[:, 1], c_x, c_y)
            inside = r_seeds < r
            r_seeds[inside] += 2.0 * (r - r_seeds[inside])
            seeds[:, 0], seeds[:, 1] = pol2cart(r_seeds, phi_seeds, c_x, c_y)

            # Reseed from inlet strip
            if np.mod(i, newseed_rate) == 0:
                idout = np.argsort(seeds[:, 0])[-newseed_num:]

                accepted = []
                while len(accepted) < newseed_num:
                    trial = rng.random([newseed_num, 3])
                    trial[:, 0] *= newseed_bw
                    trial[:, 2] = 0.0

                    if random_dist:
                        trial[:, 1] = rng.normal(c_y, std_seed, newseed_num)
                    else:
                        trial[:, 1] *= H

                    good = np.logical_and(trial[:, 1] > ymin, trial[:, 1] < ymax)
                    good &= ((trial[:, 0] - c_x) ** 2 + (trial[:, 1] - c_y) ** 2 > r ** 2)

                    accepted.extend(list(trial[good]))

                newseeds = np.array(accepted[:newseed_num])
                seeds[idout, :] = newseeds

            # --------------------------------------------------------
            # Paired snapshot storage
            # --------------------------------------------------------
            if mesh.comm.rank == 0:
                if store_first:
                    seeds_to_store1[:, :, ist1] = seeds[:, :2]
                    for ith in range(n_threads):
                        lrange = np.arange(local_range[ith][0], local_range[ith][1])
                        u_to_store1[lrange, :, ist1] = ulocal[ith]
                    ist1 += 1

                if store_second:
                    seeds_to_store2[:, :, ist2] = seeds[:, :2]
                    for ith in range(n_threads):
                        lrange = np.arange(local_range[ith][0], local_range[ith][1])
                        u_to_store2[lrange, :, ist2] = ulocal[ith]
                    ist2 += 1

            if store_first:
                flag_store = True

        progress.close()

        # ============================================================
        # Save data for post-processing
        # ============================================================
        if mesh.comm.rank == 0:
            num_stored = min(ist1, ist2)

            u_to_store1 = u_to_store1[:, :, :num_stored]
            u_to_store2 = u_to_store2[:, :, :num_stored]
            seeds_to_store1 = seeds_to_store1[:, :, :num_stored]
            seeds_to_store2 = seeds_to_store2[:, :, :num_stored]

            output_path = cfg.output_dir / f"{fname4npsave}.npz"
            print(f"Saving the result to {output_path} ...", flush=True)

            np.savez(
                output_path,
                seeds1=seeds_to_store1,
                vel1=u_to_store1,
                seeds2=seeds_to_store2,
                vel2=u_to_store2,
                xglobal=xglobal,
                C_D=C_D,
                C_L=C_L,
                t_u=t_u,
                t_p=t_p,
                p_diff=p_diff,
                centers=centers,
                R=R,
                domain=np.array([[xmin, xmax], [ymin, ymax]])
            )
    finally:
        gmsh.finalize()


def main() -> None:
    cfg = config_from_args(build_arg_parser().parse_args())
    run_simulation(cfg)


if __name__ == "__main__":
    main()
