def islandPerimeter(grid):
    circum = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                if j == 0:
                    circum += 1
                if j == len(grid[i]) - 1:
                    circum += 1
                if j - 1 >= 0 and grid[i][j-1] == 0:
                    circum += 1
                if j + 1 < len(grid[i]) and grid[i][j+1] == 0:
                    circum += 1
                if i == 0:
                    circum += 1
                if i == len(grid) - 1:
                    circum += 1
                if i - 1 >= 0 and grid[i-1][j] == 0:
                    circum += 1
                if i + 1 < len(grid) and grid[i+1][j] == 0:
                    circum += 1
    return circum

x = [[0, 1, 0, 0],
 [1, 1, 1, 0],
 [0, 1, 0, 0],
 [1, 1, 0, 0]]

res = islandPerimeter(x)
