# Sudoku Solver using Backtracking Algorithm

# To print the Grid
def Sudoku_grid(array):
    print("Solved Sudoku:")
    for a in range(9):
        print(array[a])


# To find the empty box
def empty_box(array, l):
    for row in range(9):
        for column in range(9):
            if array[row][column] == 0:
                l[0] = row
                l[1] = column
                return True
    return False


# To check if the number is repeated in the row.
def row_check(array, row, number):
    for a in range(9):
        if array[row][a] == number:
            return True
    return False


# To check if the number is repeated in the column.
def column_check(array, column, number):
    for a in range(9):
        if array[a][column] == number:
            return True
    return False


# To check if the number is repeated in the cube
def cube_check(array, row, column, number):
    for a in range(3):
        for b in range(3):
            if array[a + row][b + column] == number:
                return True
    return False


# To check for the valid number
def valid_check(array, row, column, number):
    return not row_check(array, row, number) and not column_check(array, column, number) and not cube_check(array, row - row % 3,column - column % 3, number)


# To check non-duplication across rows, columns, and boxes)
def sudoku_solve(array):
    l = [0, 0]
    if not empty_box(array, l):
        return True

# Assigning list values to row and column that we got from the above Function
    row = l[0]
    column = l[1]
    for number in range(1, 10):

        # if looks promising
        if (valid_check(array, row, column, number)):
            array[row][column] = number
            if (sudoku_solve(array)):
                return True
            array[row][column] = 0

    return False #Triggers Backtracking


# Driver main function to test above functions
if __name__ == "__main__":

    # assigning values to the grid
    grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
            [5, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 7, 0, 0, 0, 0, 3, 1],
            [0, 0, 3, 0, 1, 0, 0, 8, 0],
            [9, 0, 0, 8, 6, 3, 0, 0, 5],
            [0, 5, 0, 0, 9, 0, 6, 0, 0],
            [1, 3, 0, 0, 0, 0, 2, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 7, 4],
            [0, 0, 5, 2, 0, 6, 3, 0, 0]]

    # if success print the grid
    if sudoku_solve(grid):
        Sudoku_grid(grid)
    else:
        print("No solution exists")
