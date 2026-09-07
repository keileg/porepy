import numpy as np

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.numerics.solvers import line_search as ls
from porepy.numerics.solvers.telemetry import (
    NonlinearTelemetryRecorder,
    TelemetryConfig,
    read_jsonl_records,
)


class ConstraintFunctionsMomentumBalance(
    SquareDomainOrthogonalFractures, pp.MomentumBalance
):
    """Enhance MomentumBalance for compatibility with the solver.

    The requirement for the constraint-based line search is that the model has the
    constraint functions implemented, see ConstraintLineSearch. This class adds the
    constraint functions to the MomentumBalance model.

    The values used are unitary, so we only test that the line search is performed
    without errors.

    """

    def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Function describing the state of the opening constraint.

        Negative for open fractures, positive for closed ones.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Indicator function as an operator.

        """
        nc = sum([g.num_cells for g in subdomains])
        space = pp.ad.OperatorSpace.from_domains(
            subdomains, {pp.ad.GridEntity.cells: 1}
        )
        ind = pp.ad.DenseArray(np.ones(nc), source=space, target=space)
        return ind

    def sliding_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Function describing the state of the sliding constraint.

        Negative for sliding fractures, positive for non-sliding ones.

                Parameters:
            subdomains: List of subdomains.

        Returns:
            Indicator function as an operator.

        """
        nc = sum([g.num_cells for g in subdomains])
        space = pp.ad.OperatorSpace.from_domains(
            subdomains, {pp.ad.GridEntity.cells: 1}
        )
        ind = pp.ad.DenseArray(np.ones(nc), source=space, target=space)
        return ind


def test_line_search():
    """Test the line search on a model with constraint functions.

    The test is only to check that the line search is performed without errors. No
    checks are made on the results or that the line search does what it is supposed to.

    """
    model = ConstraintFunctionsMomentumBalance({"times_to_export": []})
    solver_params = {
        "nonlinear_solver": ls.ConstraintLineSearchNonlinearSolver,
        "global_line_search": True,
        "local_line_search": True,
    }
    pp.ModelRunner(model, solver_params).run()


def test_line_search_records_telemetry(tmp_path):
    """The line search solver must forward a recorder to the Newton loop it inherits.

    This is the only solver that produces the ``line_search`` telemetry field, since
    that field reads the ``constraint_weights`` only ``ConstraintLineSearch`` writes.
    A failure here means either that the recorder never reaches the loop, or that the
    accepted relaxation factor is no longer where the recorder looks for it.

    """
    model = ConstraintFunctionsMomentumBalance({"times_to_export": []})
    recorder = NonlinearTelemetryRecorder(
        TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
    )
    solver = ls.ConstraintLineSearchNonlinearSolver(
        {"global_line_search": True, "local_line_search": True}, telemetry=recorder
    )
    pp.ModelRunner(model, nonlinear_solver=solver).run()

    iterations = [
        record
        for record in read_jsonl_records(tmp_path / "telemetry.jsonl")
        if record["record_type"] == "iteration"
    ]
    assert iterations
    assert all(record["line_search"]["alpha"] > 0 for record in iterations)
