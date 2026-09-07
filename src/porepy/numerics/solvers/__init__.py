from __future__ import annotations

__all__ = []

from . import (
    anderson_acceleration,
    convergence_check,
    cycle_detection,
    equation_variable_tags,
    line_search,
    linear_solvers,
    newton_solver,
    nonlinear_solvers,
    sequential_nonlinear_solver,
)
from .anderson_acceleration import *
from .convergence_check import *
from .cycle_detection import *
from .equation_variable_tags import *
from .line_search import *
from .linear_solvers import *
from .linear_solvers.linear_solver import (
    LinearSolverBase,
    LinearSolverStatus,
    LinearSystem,
)
from .newton_solver import *
from .nonlinear_solvers import *
from .sequential_nonlinear_solver import *

__all__.extend(anderson_acceleration.__all__)
__all__.extend(convergence_check.__all__)
__all__.extend(cycle_detection.__all__)
__all__.extend(line_search.__all__)
__all__.extend(newton_solver.__all__)
__all__.extend(nonlinear_solvers.__all__)
__all__.extend(linear_solvers.__all__)
__all__.extend(equation_variable_tags.__all__)
__all__.extend(sequential_nonlinear_solver.__all__)

# Imported last, and deliberately not part of the tuple import above: telemetry.py
# pulls in porepy.viz.solver_statistics (for its numpy-aware JSON encoder), which in
# turn imports porepy.time_stepper.time_step_status, whose dataclass field
# annotations (no `from __future__ import annotations` there) are evaluated eagerly
# against solvers.NonlinearSolverStatus / solvers.NonlinearSolverStatusConverged --
# i.e. this package's own namespace, which is only populated by the `import *` lines
# above, not yet by the initial tuple import.
from . import telemetry
from .telemetry import *

__all__.extend(telemetry.__all__)
