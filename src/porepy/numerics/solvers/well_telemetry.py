"""State of a model's wells at a Newton iteration, for solver telemetry.

A well concentrates a model's flow in a handful of cells, and its state can change
faster than anywhere else: an opening takes its fluxes from rest to full rate within
minutes, and a column of produced fluid can heat by the whole geothermal range of its
depth. Global norms hide this, since a well has a few cells among many thousands.

This module evaluates, for every well grid, the junctions (0-d grids) coupling it to
the rest of the model, and the interfaces of both:
- every variable on these grids: its value after the iteration and its increment;
- every equation on them: its residual;
- on the well grids also the fluid density and specific enthalpy, the face fluxes, the
  type of the Darcy flux condition on faces at the domain boundary (the wellhead), and
  the cell-wise CFL number.

Everything is in simulation units. The grids are small, so the cost is a few operator
evaluations per well and iteration.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

__all__ = ["WellTopology", "find_well_topology", "well_state_fields", "cell_cfl"]


@dataclass
class WellTopology:
    """The grids whose state well telemetry records."""

    wells: list[pp.Grid]
    """The 1-d subdomains for which the model's `is_well_grid` holds."""
    junctions: list[pp.Grid]
    """0-d subdomains coupled to a well, e.g. where a well meets a fracture."""
    interfaces: list[pp.MortarGrid]
    """Interfaces of the wells and of the junctions."""


def find_well_topology(model: pp.PorePyModel) -> Optional[WellTopology]:
    """Find the wells of a model, their junctions and the interfaces of both.

    Parameters:
        model: A prepared model.

    Returns:
        The topology, or `None` for a model without wells.

    """
    if not hasattr(model, "is_well_grid"):
        return None
    mdg = model.mdg
    wells = [sd for sd in mdg.subdomains(dim=1) if model.is_well_grid(sd)]
    if not wells:
        return None

    def other_side(interface: pp.MortarGrid, subdomain: pp.Grid) -> pp.Grid:
        primary, secondary = mdg.interface_to_subdomain_pair(interface)
        return secondary if primary is subdomain else primary

    junctions: list[pp.Grid] = []
    interfaces: list[pp.MortarGrid] = []
    for well in wells:
        for interface in mdg.subdomain_to_interfaces(well):
            interfaces.append(interface)
            neighbour = other_side(interface, well)
            if neighbour.dim == 0 and neighbour not in junctions:
                junctions.append(neighbour)
    for junction in junctions:
        for interface in mdg.subdomain_to_interfaces(junction):
            if interface not in interfaces:
                interfaces.append(interface)
    return WellTopology(wells=wells, junctions=junctions, interfaces=interfaces)


def well_state_fields(
    model: pp.PorePyModel,
    topology: WellTopology,
    nonlinear_increment: np.ndarray,
    residual: np.ndarray,
    equation_indexer: pp.ad.EquationIndexer,
    variable_indexer: pp.ad.VariableIndexer,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate the state of the wells, their junctions and interfaces.

    Parameters:
        model: The model being solved, holding the iterate after the increment.
        topology: The model's wells, from :func:`find_well_topology`.
        nonlinear_increment: The increment, laid out as `variable_indexer`.
        residual: The residual after the increment, laid out as `equation_indexer`.
        equation_indexer: Indexer scoped to `residual`.
        variable_indexer: Indexer scoped to `nonlinear_increment`.

    Returns:
        Lists of per-grid fields under ``wells``, ``junctions`` and ``interfaces``. Each
        holds the grid's id and, per variable on it, its ``value`` and ``increment``
        (`None` for a variable the solver does not update), and per equation on it, its
        residual. Wells add their geometry, fluid properties, face fluxes, wellhead
        conditions and CFL numbers; junctions their depth; interfaces their
        codimension and the ids of the two subdomains they couple.

    """
    increments: dict[tuple[str, pp.Grid | pp.MortarGrid], np.ndarray] = {
        (variable.name, variable.domain): nonlinear_increment[indices]
        for variable, indices in variable_indexer.indices.items()
    }
    residuals = {
        (equation.name, equation.domain): residual[indices]
        for equation, indices in equation_indexer.indices.items()
    }

    def grid_fields(grid: pp.Grid | pp.MortarGrid) -> dict[str, Any]:
        """Variables and residuals of one grid."""
        return {
            "variables": _variable_fields(model, grid, increments),
            "residuals": {
                name: values.tolist()
                for (name, domain), values in residuals.items()
                if domain is grid
            },
        }

    mdg = model.mdg
    return {
        "wells": [
            {
                "domain_id": well.id,
                "well_num": int(well.well_num),
                "well_name": _well_name(model, well),
                "cell_z": well.cell_centers[2].tolist(),
                "face_z": well.face_centers[2].tolist(),
                **grid_fields(well),
                **_well_flow_fields(model, well),
            }
            for well in topology.wells
        ],
        "junctions": [
            {
                "domain_id": junction.id,
                "z": float(junction.cell_centers[2, 0]),
                **grid_fields(junction),
            }
            for junction in topology.junctions
        ],
        "interfaces": [
            {
                "interface_id": interface.id,
                "codim": interface.codim,
                "subdomain_ids": [
                    sd.id for sd in mdg.interface_to_subdomain_pair(interface)
                ],
                **grid_fields(interface),
            }
            for interface in topology.interfaces
        ],
    }


def cell_cfl(
    grid: pp.Grid,
    mass_flux: np.ndarray,
    density: np.ndarray,
    pore_volume: np.ndarray,
    dt: float,
) -> np.ndarray:
    """CFL number of each cell: the volume of fluid passing through it in one time
    step, relative to its pore volume.

    The flow through a cell is the larger of its total inflow and total outflow over
    its faces, so that a cell where flow converges or diverges counts the same as one
    it passes through. For a well, the faces include those coupled to junctions, whose
    fluxes carry the interface flux.

    Parameters:
        grid: The grid the fluxes are defined on.
        mass_flux: Mass flux on each face, positive along the face normal.
        density: Fluid density in each cell.
        pore_volume: Pore volume of each cell.
        dt: Time step size.

    Returns:
        The CFL number of each cell.

    """
    # cell_faces holds +1 where the face normal points out of the cell, so a positive
    # product is outflow.
    signed = sps.csc_matrix(grid.cell_faces.multiply(mass_flux[:, np.newaxis]))
    outflow = np.asarray(signed.maximum(0).sum(axis=0)).ravel()
    inflow = np.asarray((-signed).maximum(0).sum(axis=0)).ravel()
    return dt * np.maximum(outflow, inflow) / (density * pore_volume)


def _variable_fields(
    model: pp.PorePyModel,
    grid: pp.Grid | pp.MortarGrid,
    increments: dict[tuple[str, pp.Grid | pp.MortarGrid], np.ndarray],
) -> dict[str, dict[str, Optional[list[float]]]]:
    """Value and increment of every variable on a grid."""
    equation_system = model.equation_system
    fields: dict[str, dict[str, Optional[list[float]]]] = {}
    for variable in equation_system.get_variables(grids=[grid]):
        increment = increments.get((variable.name, grid))
        fields[variable.name] = {
            "value": equation_system.get_variable_values(
                variables=[variable], iterate_index=0
            ).tolist(),
            "increment": None if increment is None else increment.tolist(),
        }
    return fields


def _well_name(model: pp.PorePyModel, well: pp.Grid) -> Optional[str]:
    """The ``well_name`` tag of the well a grid belongs to, if the model has one."""
    network = getattr(model, "well_network", None)
    if network is None or not 0 <= well.well_num < len(network.wells):
        return None
    return network.wells[well.well_num].tags.get("well_name")


def _well_flow_fields(model: pp.PorePyModel, well: pp.Grid) -> dict[str, Any]:
    """Fluid properties, face fluxes, wellhead conditions and CFL numbers of a well.

    Each quantity the model does not define is `None`, e.g. the enthalpy flux of a
    model without an energy balance.

    """

    def evaluate(operator: Callable[[list[pp.Grid]], pp.ad.Operator]) -> np.ndarray:
        values = model.equation_system.evaluate(operator([well]))
        return np.atleast_1d(np.asarray(values, dtype=float))

    def evaluate_if_defined(owner: Any, name: str) -> Optional[np.ndarray]:
        operator = getattr(owner, name, None)
        if operator is None:
            return None
        try:
            return evaluate(operator)
        except NotImplementedError:
            # A fluid property declared without a constitutive law mixed in, such as
            # the enthalpy of a model without an energy balance.
            return None

    fluid = getattr(model, "fluid", None)
    density = evaluate_if_defined(fluid, "density")
    specific_enthalpy = evaluate_if_defined(fluid, "specific_enthalpy")
    faces = {
        name: evaluate_if_defined(model, name)
        for name in ("darcy_flux", "fluid_flux", "enthalpy_flux", "fourier_flux")
    }
    return {
        "fluid": {
            "density": _as_list(density),
            "specific_enthalpy": _as_list(specific_enthalpy),
        },
        "faces": {name: _as_list(values) for name, values in faces.items()},
        "wellhead": _wellhead_fields(model, well),
        "cfl": _as_list(_well_cfl(model, well, faces["fluid_flux"], density)),
    }


def _well_cfl(
    model: pp.PorePyModel,
    well: pp.Grid,
    mass_flux: Optional[np.ndarray],
    density: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """CFL numbers of a well's cells, or `None` if the model lacks what they need."""
    time_manager = getattr(model, "time_manager", None)
    if (
        mass_flux is None
        or density is None
        or time_manager is None
        or not hasattr(model, "porosity")
        or not hasattr(model, "specific_volume")
    ):
        return None
    porosity = np.asarray(model.equation_system.evaluate(model.porosity([well])))
    specific_volume = np.asarray(
        model.equation_system.evaluate(model.specific_volume([well]))
    )
    pore_volume = (
        np.broadcast_to(porosity, well.num_cells)
        * np.broadcast_to(specific_volume, well.num_cells)
        * well.cell_volumes
    )
    return cell_cfl(well, mass_flux, density, pore_volume, time_manager.dt)


def _wellhead_fields(
    model: pp.PorePyModel, well: pp.Grid
) -> Optional[dict[str, list[Any]]]:
    """The well's faces on the domain boundary, and the type of their Darcy flux
    condition; `None` for a model without one."""
    if not hasattr(model, "bc_type_darcy_flux"):
        return None
    faces = np.flatnonzero(well.tags["domain_boundary_faces"])
    condition = model.bc_type_darcy_flux(well)
    types = np.where(
        condition.is_dir[faces], "dir", np.where(condition.is_rob[faces], "rob", "neu")
    )
    return {"faces": faces.tolist(), "darcy_flux_bc": types.tolist()}


def _as_list(values: Optional[np.ndarray]) -> Optional[list[float]]:
    return None if values is None else values.tolist()
