import itertools as it
import json
from collections import defaultdict

try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text


def load_cages(path, verbose=False):
    """
    Loads and validates Sudoku cages from a JSON file.

    Args:
        path (str): The file path to the JSON file containing the cages.
        verbose (bool, optional): If True, prints detailed information during loading. Defaults to False.

    Returns:
        list: A list of dictionaries, each representing a cage with the following keys:
            - "cells" (list): A list of cell identifiers in the cage.
            - "sum" (int): The sum of the numbers in the cage.
            - "max_number" (int): The maximum possible sum for the cage based on its size.

    Raises:
        ValueError: If any cage has less than 2 cells, an invalid sum, or contains invalid or repeated cells.
    """
    used_cells = set()
    cages = []
    valid_cells = {f"{col}{row}": set(range(1, 10))
                   for row, col in it.product("123456789", "ABCDEFGHI")}
    with open(path) as board_file:
        raw_data = json.load(board_file)
        for cage in raw_data:
            cells = cage["cells"]
            sum = cage["sum"]
            if verbose:
                print(f"Loading cage {cells} with sum {sum}")
            # Check if the cage is valid
            if len(cells) < 2:
                print(f"Error: Cage {cells} has less than 2 cells.")
                return
            # S = Sb - Sa-1
            max_number = 45 - (9 - len(cells)) * (9 - len(cells) + 1) // 2
            if sum < 3 or sum > max_number:
                print(f"Error: Cage {cells} has an invalid sum.")
                return
            # Check if the cells are valid
            for cell in cells:
                if cell not in valid_cells:
                    print(f"Error: Cell {cell} is not valid.")
                    return
                if cell in used_cells:
                    print(f"Error: Cell {cell} is repeated.")
                    return
                used_cells.add(cell)
            cage = {
                "cells": cells,
                "sum": sum,
                "max_number": max_number
            }
            if verbose:
                print(f"Cage {cage}")
            cages.append(cage)
    return cages


def find_possible_cell_combinations_for_cage(varsValues, cells, target_sum):
    """
    Finds all possible combinations of values for the cells in a cage that sum up to the target sum.

    Args:
        varsValues (dict): Dictionary of possible values for each cell.
        cells (list): List of cell identifiers in the cage.
        target_sum (int): The target sum to achieve by selecting one value for each cell.

    Yields:
        tuple: A tuple of values, one for each cell in the cage, that sum up to the target sum.
    """

    if len(cells) == 1:
        if target_sum in varsValues[cells[0]]:
            yield (target_sum,)
        return

    # Find all possible combinations of values that sum up to the target sum
    cells = cells.copy()
    chosen_cell = cells.pop()

    for value in varsValues[chosen_cell]:
        for combination in find_possible_cell_combinations_for_cage(varsValues, cells, target_sum - value):
            if all(value != val for val in combination):
                yield combination + (value,)

class Stats(object):
    def __init__(self):
        self.iteration_counter = 0
        self.look_forward_stat = 0
        self.no_repetition_stat = 0
        self.naked_subsets_stat = 0
        self.hidden_single_stat = 0
        self.cage_sum_stat = 0

class SudokuKillerSolver:
    def __init__(self, cages, verbose=False):
        self.verbose = verbose
        self.cages = cages
        cols = "ABCDEFGHI"
        rows = "123456789"
        self.stats = Stats()
        self.variables = {f"{col}{row}": set(range(1, 10))
                          for row, col in it.product("123456789", "ABCDEFGHI")}
        self.unique_groups = [
            # Rows
            [f"{col}{row}" for col in cols] for row in rows
        ] + [
            # Columns
            [f"{col}{row}" for row in rows] for col in cols
        ] + [
            # 3x3 Blocks
            [f"{cols[c + dc]}{rows[r + dr]
                              }" for dr in range(3) for dc in range(3)]
            for r in range(0, 9, 3)
            for c in range(0, 9, 3)
        ]

    def apply_no_repetition_constraint(self):
        """
        Applies the no repetition constraint to the Sudoku board.

        The no repetition constraint ensures that each value appears only once
        in each row, column, and 3x3 block. If a cell has a single possible value,
        that value is removed from the possible values of other cells in the same group.

        Returns:
            bool: True if any changes were made to the board, False otherwise.

        Raises:
            ValueError: If a cell is left without possible values after applying the constraint.
        """
        changed = False
        for group in self.unique_groups:
            for cell in group:
                if len(self.variables[cell]) == 1:
                    value = next(iter(self.variables[cell]))
                    for other_cell in group:
                        if other_cell != cell:
                            if value in self.variables[other_cell]:
                                self.variables[other_cell].discard(value)
                                self.stats.no_repetition_stat += 1
                                if len(self.variables[other_cell]) == 0:
                                    raise ValueError(
                                        f"Error: Given {cell} = {value}, {other_cell} is left without possible values.")
                                changed = True
                                if self.verbose:
                                    print(f"Removing {value} from {
                                          other_cell} due {cell} = {value}, {self.variables[other_cell]} remaining")
        return changed

    def apply_naked_subsets_constraint(self):
        """
        Applies the naked subsets constraint to the Sudoku board.

        The naked subsets constraint identifies groups of cells within a row, column, or block
        that share the same set of possible values, where the number of cells is equal to the number
        of possible values. When such a group is found, those values can be removed from the possible
        values of other cells in the same group.

        Returns:
            bool: True if any changes were made to the board, False otherwise.

        Raises:
            ValueError: If a cell is left without possible values after applying the constraint.
        """
        changed = False
        for constraint in self.unique_groups:
            # Segregate cells by the number of possible values
            shared_length = [[] for _ in range(7)]  # 2 to 8
            for cell in constraint:
                values = self.variables[cell]
                if 1 < len(values) <= 8:
                    shared_length[len(values) -
                                  2].append((cell, frozenset(values)))
            for group in shared_length:
                if len(group) == 0:
                    continue
                shared_values = defaultdict(list)
                for cell, values in group:
                    shared_values[values].append(cell)
                for values, cells in shared_values.items():
                    if len(cells) != len(values):
                        continue
                    # cells and values represent a group of cells that share the same set of possible values,
                    # and the number of cells is equal to the number of possible values, as all the cells are
                    # in the same group, so the remaining cells in the group cannot have the same values
                    for cell in constraint:
                        if cell in cells:
                            continue
                        for value in values:
                            if value in self.variables[cell]:
                                self.variables[cell].discard(value)
                                if len(self.variables[cell]) == 0:
                                    raise ValueError(
                                        f"Error: Cells {cells} share the same possible values {values}, leaving {cell} without possible values.")
                                changed = True
                                self.stats.naked_subsets_stat += 1
                                if self.verbose:
                                    print(
                                        f"Removing {value} from {cell} due to naked subsets constraint, {self.variables[cell]} remaining")

        return changed

    def hidden_single_constraint(self):
        """
        Applies the hidden single constraint to the Sudoku board.

        The hidden single constraint identifies cells in a group (row, column, or block)
        that are the only possible location for a particular value. When such a cell is found,
        it is assigned that value, and the value is removed from the possible values of other cells
        in the group.

        Returns:
            bool: True if any changes were made to the board, False otherwise.
        """
        changed = False
        for group in self.unique_groups:
            values_by_cells = [[] for _ in range(9)]
            for cell in group:
                for value in self.variables[cell]:
                    values_by_cells[value - 1].append(cell)
            for i, cells_with_this_value in enumerate(values_by_cells):
                if len(cells_with_this_value) == 1 and len(self.variables[cells_with_this_value[0]]) > 1:
                    value = i + 1
                    hidden_single = cells_with_this_value[0]
                    # As hidden_single is the unique cell that can have value, we can remove all other values from it
                    self.variables[hidden_single] = {value}
                    changed = True
                    self.stats.hidden_single_stat += 1
                    if self.verbose:
                        print(
                            f"Assigning {value} to {hidden_single} due to hidden single constraint")
        return changed

    def apply_cage_sum_constraint(self):
        """
        Applies the cage sum constraint to the Sudoku board.

        The cage sum constraint ensures that the sum of the values in each cage
        matches the target sum specified for that cage. It also updates the possible
        values for each cell in the cage based on the valid combinations of values
        that can achieve the target sum.

        Returns:
            bool: True if any changes were made to the board, False otherwise.

        Raises:
            ValueError: If a cage has no possible combinations of values that sum up to the target sum.
        """
        changed = False
        for cage in self.cages:
            combinations = list(find_possible_cell_combinations_for_cage(
                self.variables, cage['cells'], cage['sum']))
            if len(combinations) == 0:
                raise ValueError(
                    f"Error: Cage {cage['cells']} has no possible combinations of values that sum up to {cage['sum']}.")
            for cell, valid_values in zip(cage['cells'], zip(*combinations)):
                if self.variables[cell] != set(valid_values):
                    removed_values = self.variables[cell] - set(valid_values)
                    self.variables[cell] = set(valid_values)
                    self.stats.cage_sum_stat += len(removed_values)
                    if self.verbose:
                        print(
                            f"Removing {removed_values} from {cell} due to cage sum constraint, {self.variables[cell]} remaining")
                    changed = True

        return changed

    def apply_all_constraints(self):
        for _ in range(1): # look forward has demonstrated to be more efficient than multiple iterations of constraints
            changed = False
            changed |= self.apply_no_repetition_constraint()
            changed |= self.apply_naked_subsets_constraint()
            changed |= self.hidden_single_constraint()
            changed |= self.apply_cage_sum_constraint()
            self.stats.iteration_counter += 1
            if not changed:
                break

    def look_forward(self):
        """
        Look-forward algorithm to solve the Sudoku by assigning values and propagating constraints.
        The algorithm takes a recursive approach to solve the board, backtracking if a dead-end is reached.
        At the beginning of each step, a cell is chosen heuristically, and from it, the algorithm tries to find a 
        solution by testing each possible value for that cell. If a value is found to be valid for the current state 
        of the board, the algorithm proceeds to the next cell, and so on. If a cell is left without possible values,
        the algorithm backtracks to the last decision where the path was viable and tries the next possible value.
        """

        unassigned_vars = [
            var for var in self.variables if len(self.variables[var]) > 1]
        if not unassigned_vars:
            return True

        # Heuristic: find the cage closest to being solved and its cell with the fewest possible values
        chosen = min(unassigned_vars, key=lambda v: len(self.variables[v]))

        if verbose:
            print(f"\nSelecting {chosen} with possible values: {
                self.variables[chosen]}. {len(unassigned_vars)} unassigned cells remaining.")

        # Evaluate the viability of assigning each possible value to the chosen cell
        for value in self.variables[chosen].copy():
            if verbose:
                print(f"Trying to assign {chosen} = {value}")

            # Deep copy the board dictionary to rollback changes if needed
            rollback_variables = {k: v.copy()
                                  for k, v in self.variables.items()}
            rollback_stats = self.stats.__dict__.copy()

            # Assign the value to the chosen cell
            self.variables[chosen] = {value}

            try:
                # Apply constraints to propagate the effect of the assignment
                self.apply_all_constraints()
            except ValueError as e:
                # A constraint was violated, so the current value is invalid
                if verbose:
                    print(e)
                self.variables = rollback_variables
                self.stats.__dict__.update(rollback_stats)
                continue

            # Recursively try to solve the board
            if self.look_forward():
                self.stats.look_forward_stat += 1
                return True
            else:
                # Backtrack
                self.variables = rollback_variables
                self.stats.__dict__.update(rollback_stats)

        return False

    def solve(self):
        if self.look_forward():
            self.print_board()
            print("Sudoku solved!")
        else:
            print("No solution found.")

    def calculate_progress(self):
        """
        Calculates the progress of solving the Sudoku board.

        The progress is calculated as the proportion of known values,
        where each additional possible value for a cell is considered an unknown.

        Returns:
            float: The progress of solving the Sudoku board.
        """
        unknowns = sum(len(values) - 1 for values in self.variables.values())
        return 1 - unknowns / (9 * 9 * (9 - 1))

    def print_board(self):
        # Define colors for cages
        colors = ['red', 'green', 'yellow', 'blue',
                  'magenta', 'cyan', 'white', 'dark_grey', 'light_red', 'light_green', 'light_yellow', 'light_blue', 'light_magenta', 'light_cyan']
        cage_colors = {}
        for i, cage in enumerate(self.cages):
            for cell in cage['cells']:
                cage_colors[cell] = colors[i % len(colors)]

        horizontal_border = "╔═══════╦═══════╦═══════╗"
        middle_border = "╠═══════╬═══════╬═══════╣"
        bottom_border = "╚═══════╩═══════╩═══════╝"
        row_separator = "║"

        print(horizontal_border)
        for i, row in enumerate("123456789"):
            print(row_separator, end="")
            for col in "ABCDEFGHI":
                cell_values = list(self.variables[f"{col}{row}"])
                if len(cell_values) == 1:
                    value = str(cell_values[0])
                elif len(cell_values) == 0:
                    value = '!'
                else:
                    value = '.'
                if col == 'D' or col == 'G':
                    print(" " + row_separator, end="")
                if f"{col}{row}" in cage_colors:
                    print(
                        colored(" " + value, cage_colors[f"{col}{row}"]), end="")
                else:
                    print(value, end="")
            print(" " + row_separator)
            if i == 2 or i == 5:
                print(middle_border)
        print(bottom_border)
        for i, cage in enumerate(self.cages):
            print(f"Cage {colored("█", colors[i % len(colors)])} {
                  cage['cells']}: {cage['sum']}")


if __name__ == "__main__":
    verbose = False
    cages = load_cages("KL5TDHPN", verbose)
    solver = SudokuKillerSolver(cages, verbose)
    solver.solve()
    if verbose:
        print(f"Solved in {solver.stats.iteration_counter} iterations.")
        print(f"{solver.stats.look_forward_stat} values reduced by look-forward.")
        print(f"{solver.stats.no_repetition_stat} values reduced by no repetition.")
        print(f"{solver.stats.naked_subsets_stat} values reduced by naked subsets.")
        print(f"{solver.stats.hidden_single_stat} values reduced by hidden single.")
        print(f"{solver.stats.cage_sum_stat} values reduced by cage sum.")
