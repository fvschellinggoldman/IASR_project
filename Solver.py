import numpy as np


class SudokuSolver():
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.puzzle_o = np.array(puzzle, copy=True)

    def grid_ref(self, number):
        grid_ref = (number // 9, number % 9)
        return grid_ref

    def value(self, grid, number):
        g_r = self.grid_ref(number)
        value = grid[g_r]
        return value

    def cell(self, grid, number):
        g_r = self.grid_ref(number)
        cell_ref = (g_r[0] // 3, g_r[1] // 3)
        cell = grid[((cell_ref[0]) * 3):((cell_ref[0]) * 3) + 3, ((cell_ref[1]) * 3):((cell_ref[1]) * 3) + 3]
        return cell

    def row(self, grid, number):
        g_r = self.grid_ref(number)
        row_ref = g_r[0]
        row = grid[(row_ref):(row_ref + 1), 0:9]
        return row

    def column(self, grid, number):
        g_r = self.grid_ref(number)
        column_ref = g_r[1]
        column = grid[0:9, (column_ref):(column_ref + 1)]
        return column

    def solvePuzzle(self):
        forwards = True
        i = 0
        while i < 9 * 9:
            if self.value(self.puzzle_o, i) == 0 and forwards:
                for a in range(1, 10):
                    if a not in self.cell(self.puzzle, i) and a not in self.row(self.puzzle,
                                                                                i) and a not in self.column(self.puzzle,
                                                                                                            i):
                        self.puzzle[self.grid_ref(i)] = a
                        i += 1
                        break
                    else:
                        if a == 9:
                            forwards = False
                            i -= 1  # goes back a cell
                            break
            elif self.value(self.puzzle_o, i) != 0 and forwards:
                i += 1
            elif self.value(self.puzzle_o, i) == 0 and not forwards:
                if self.puzzle[self.grid_ref(i)] == 9:
                    self.puzzle[self.grid_ref(i)] = 0
                    i -= 1
                else:
                    for a in range(self.puzzle[self.grid_ref(i)] + 1, 10):
                        if a not in self.cell(self.puzzle, i) and a not in self.row(self.puzzle,
                                                                                    i) and a not in self.column(
                                self.puzzle, i):
                            self.puzzle[self.grid_ref(i)] = a
                            forwards = True
                            i += 1
                            break
                        else:
                            if a == 9:
                                self.puzzle[self.grid_ref(i)] = 0
                                i -= 1
                                break
            elif self.value(self.puzzle_o, i) != 0 and not forwards:
                i -= 1
        return self.puzzle
