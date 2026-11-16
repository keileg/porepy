"""Tests of the well state recorded by solver telemetry.

The fixture model has one vertical well crossing a horizontal fracture, so it has the
three kinds of grid the module covers: a well, a 0-d junction where the well meets the
fracture, and the interfaces coupling the junction to the well (codimension 1) and to
the fracture (codimension 2).

"""

from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import CubeDomainOrthogonalFractures
from porepy.applications.test_utils import well_models
from porepy.numerics.solvers.well_telemetry import (
    WellTopology,
    _on_cells,
    cell_cfl,
    find_well_topology,
    well_state_fields,
)


class _WellGeometry(
    well_models.OneVerticalWell,
    CubeDomainOrthogonalFractures,
    well_models.BoundaryConditionsWellSetup,
    well_models.WellPermeability,
):
    def meshing_arguments(self) -> dict:
        return {"cell_size": 0.25}

    def grid_type(self) -> Literal["cartesian"]:
        return "cartesian"


class WellModel(_WellGeometry, pp.MassAndEnergyBalance):
    """Mass and energy balance with a well crossing a fracture."""


class MassOnlyWellModel(_WellGeometry, pp.SinglePhaseFlow):
    """As `WellModel`, without an energy balance."""


def _prepared(model_class: type) -> pp.PorePyModel:
    # Fracture 2 is horizontal, at half depth, so the vertical well crosses it.
    model = model_class({"fracture_indices": [2], "times_to_export": []})
    model.prepare_simulation()
    return model


@pytest.fixture(scope="module")
def well_model() -> pp.PorePyModel:
    return _prepared(WellModel)


@pytest.fixture(scope="module")
def topology(well_model: pp.PorePyModel) -> WellTopology:
    topology = find_well_topology(well_model)
    assert topology is not None
    return topology


def _fields(
    model: pp.PorePyModel,
    topology: WellTopology,
    increment: np.ndarray,
    variable_indexer: pp.ad.VariableIndexer,
) -> dict:
    equation_system = model.equation_system
    return well_state_fields(
        model,
        topology,
        nonlinear_increment=increment,
        residual=equation_system.assemble(evaluate_jacobian=False),
        equation_indexer=equation_system.equation_indexer,
        variable_indexer=variable_indexer,
    )


def test_topology(well_model: pp.PorePyModel, topology: WellTopology) -> None:
    """The well, its junction with the fracture, and exactly the interfaces of these
    two are found. A failure most likely means a junction or an interface is missed,
    whose state would then be absent from every record.

    """
    mdg = well_model.mdg
    assert [sd.dim for sd in topology.wells] == [1]
    assert [sd.dim for sd in topology.junctions] == [0]
    recorded = set(topology.wells + topology.junctions)
    expected = {
        intf
        for intf in mdg.interfaces()
        if recorded & set(mdg.interface_to_subdomain_pair(intf))
    }
    assert set(topology.interfaces) == expected
    assert sorted(intf.codim for intf in topology.interfaces) == [1, 2]


def test_model_without_wells_has_no_topology() -> None:
    model = pp.SinglePhaseFlow({"times_to_export": []})
    model.prepare_simulation()
    assert find_well_topology(model) is None


def test_cell_cfl() -> None:
    """The CFL number counts the larger of a cell's inflow and outflow, so a cell
    where flow converges counts its total inflow. A failure most likely means the
    face orientation is applied with the wrong sign.

    """
    grid = pp.CartGrid(np.array([3]))
    grid.compute_geometry()
    # Flow enters at both ends and converges on the middle cell.
    mass_flux = np.array([1.0, 1.0, -1.0, -1.0])

    cfl = cell_cfl(
        grid, mass_flux, density=np.full(3, 2.0), pore_volume=np.full(3, 0.5), dt=3.0
    )

    assert np.allclose(cfl, [3.0, 6.0, 3.0])


def test_values_increments_and_residuals(
    well_model: pp.PorePyModel, topology: WellTopology
) -> None:
    """Each grid reports its own slice of the increment and of the residual. A failure
    most likely means a block is attributed to the wrong grid, so that the record
    describes a well with another grid's state.

    """
    equation_system = well_model.equation_system
    increment = np.arange(equation_system.num_dofs(), dtype=float)
    residual = equation_system.assemble(evaluate_jacobian=False)

    fields = _fields(well_model, topology, increment, equation_system.variable_indexer)

    well = topology.wells[0]
    recorded = fields["wells"][0]
    assert recorded["domain_id"] == well.id
    for variable in equation_system.get_variables(grids=[well]):
        indices = equation_system.dofs_of([variable])
        assert recorded["variables"][variable.name] == {
            "value": equation_system.get_variable_values(
                variables=[variable], iterate_index=0
            ).tolist(),
            "increment": increment[indices].tolist(),
        }
    for equation, indices in equation_system.equation_indexer.indices.items():
        if equation.domain is well:
            assert recorded["residuals"][equation.name] == residual[indices].tolist()
    assert recorded["cell_z"] == well.cell_centers[2].tolist()
    assert recorded["wellhead"]["darcy_flux_bc"] == ["neu"]
    assert len(recorded["cfl"]) == well.num_cells
    assert [j["domain_id"] for j in fields["junctions"]] == [
        j.id for j in topology.junctions
    ]


def test_fracture_cells_holding_a_junction(
    well_model: pp.PorePyModel, topology: WellTopology
) -> None:
    """The junction-fracture interface records the fracture cells it couples to,
    with their own slices of the variables and residuals. A failure most likely means
    the cells are taken from the wrong side of the interface, or a multi-component
    variable (contact traction, say) is sliced per entry instead of per cell.

    """
    equation_system = well_model.equation_system
    increment = np.arange(equation_system.num_dofs(), dtype=float)
    residual = equation_system.assemble(evaluate_jacobian=False)

    fields = _fields(well_model, topology, increment, equation_system.variable_indexer)

    (recorded,) = [i for i in fields["interfaces"] if i["codim"] == 2]
    interface = next(intf for intf in topology.interfaces if intf.codim == 2)
    fracture, junction = well_model.mdg.interface_to_subdomain_pair(interface)
    cells = recorded["primary_cells"]["cells"]
    assert recorded["primary_cells"]["domain_id"] == fracture.id
    # The junction lies on the fracture cells recorded for it.
    assert np.allclose(fracture.cell_centers[2, cells], junction.cell_centers[2, 0])
    pressure = equation_system.get_variables(["pressure"], [fracture])
    pressure_dofs = equation_system.dofs_of(pressure)[cells]
    assert recorded["primary_cells"]["variables"]["pressure"]["increment"] == (
        increment[pressure_dofs].tolist()
    )
    mass_balance = next(
        indices
        for equation, indices in equation_system.equation_indexer.indices.items()
        if equation.name == "mass_balance_equation" and equation.domain is fracture
    )
    assert recorded["primary_cells"]["residuals"]["mass_balance_equation"] == (
        residual[mass_balance[cells]].tolist()
    )
    assert "primary_cells" not in next(
        i for i in fields["interfaces"] if i["codim"] == 1
    )


def test_cell_slice_keeps_all_components_of_a_cell() -> None:
    """Cell-wise values are laid out cell-major, so the slice for some cells takes
    each cell's components together, as for a contact traction with three per cell.

    """
    values = np.arange(12.0)  # Four cells, three components each.
    assert _on_cells(values, np.array([1, 3]), 3).tolist() == [3, 4, 5, 9, 10, 11]
    assert _on_cells(values, None, 3) is values


def test_restricted_solver_records_no_increment_for_inactive_variables(
    well_model: pp.PorePyModel, topology: WellTopology
) -> None:
    """A solver restricted to some variables leaves the others' increments empty
    rather than filled from a misaligned increment vector.

    """
    equation_system = well_model.equation_system
    temperatures = equation_system.get_variables(["temperature"])
    restricted = pp.ad.construct_matching_restricted_indexer(
        equation_system.variable_indexer, temperatures
    )
    increment = np.ones(equation_system.dofs_of(temperatures).size)

    fields = _fields(well_model, topology, increment, restricted)

    variables = fields["wells"][0]["variables"]
    assert variables["pressure"]["increment"] is None
    assert variables["temperature"]["increment"] == [1.0] * topology.wells[0].num_cells


def test_model_without_energy_balance() -> None:
    """Quantities the model does not define are recorded as missing, not as an error,
    since the recording is on by default for every model with wells.

    """
    model = _prepared(MassOnlyWellModel)
    topology = find_well_topology(model)
    assert topology is not None
    increment = np.zeros(model.equation_system.num_dofs())

    fields = _fields(model, topology, increment, model.equation_system.variable_indexer)

    well = fields["wells"][0]
    assert set(well["variables"]) == {"pressure"}
    assert well["fluid"]["specific_enthalpy"] is None
    assert well["faces"]["enthalpy_flux"] is None
    assert well["faces"]["darcy_flux"] is not None
