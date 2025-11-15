from Bot import *

class Bot_2131850(Bot):
    def __init__(self, settings):
        super().__init__(settings)
        self.setName('Magic Sweeper')	
        self.direction = RIGHT
        self.resumeCell = []
        self.resumeRow = []
	
    def nextMove(self, currentCell, currentEnergy, vision, remainingStainCells):
        #To go up and down to clean stains
        for i in range(3):
            for j in range(3):
                if vision[i][j] == "@":
                    if self.resumeCell == []:
                        self.resumeCell = [currentCell[0], currentCell[1]]
                    if i == 0 and j == 1:
                        return UP
                    elif i == 2 and j == 1:
                        return DOWN

        # To avoid the obstacles but ignore wall
        if self.nrPillars != 0:
            for a in range(3):
                for b in range(3):
                    if vision[a][b] == "x":
                        if currentCell[0] != 1 and currentCell[0] != self.nrRows - 2 and currentCell[1] != 1 and currentCell[1] < self.nrCols - 3:
                            if not self.resumeRow:
                                if self.direction == RIGHT:
                                    if a == 1 and b == 2:
                                        self.resumeRow = [currentCell[0], currentCell[1]]
                                        return DOWN
                                elif self.direction == LEFT:
                                    if a == 1 and b == 0:
                                        self.resumeRow = [currentCell[0], currentCell[1]]
                                        return DOWN
                        # When near right wall and sees an obstacle to the left, go down until clear
                        elif currentCell[1] >= self.nrCols - 3:
                            if vision[1][0] == "x":
                                return DOWN
                            elif vision[1][2] == "x" and vision[1][0] != "x":
                                pass
                            else:
                                self.direction = LEFT
                                return LEFT
                        elif currentCell[1] <= 2:
                            if vision[1][2] == "x":
                                return DOWN
                            elif vision[1][0] == "x" and vision[1][2] != "x":
                                pass
                            else:
                                self.direction = RIGHT
                                return RIGHT
                        
        
        #To go back the cell before avoiding obstacles and continue to run
        if self.resumeRow:   
            # print(self.resumeRow)
            if self.direction == RIGHT:
                if currentCell[1] < self.resumeRow[1] + self.sizePillars + 1:
                    return RIGHT
                else:
                    self.resumeRow = []
                    return UP
            if self.direction == LEFT:
                if currentCell[1] > self.resumeRow[1] - self.sizePillars - 1:
                    return LEFT
                else:
                    self.resumeRow = []
                    return UP
            else:
                self.resumeRow = []
                return DOWN
                
        #To go back the cell before cleaning and continue to run
        if self.resumeCell:
            if [currentCell[0], currentCell[1]] == self.resumeCell:
                self.resumeCell = []  
            else:
                if currentCell[0] < self.resumeCell[0]:
                    return DOWN
                elif currentCell[0] > self.resumeCell[0]:
                    return UP
                elif currentCell[1] < self.resumeCell[1]:
                    return RIGHT
                elif currentCell[1] > self.resumeCell[1]:
                    return LEFT
                    
        #Running logic of skipping 2 lines
        if remainingStainCells != 0:
            # print(self.direction)
            if currentCell[0] == 1:
                return DOWN
            if currentCell[0] % 3 == 2:
                if self.direction == RIGHT:
                    if currentCell[1] != self.nrCols - 2:
                        return RIGHT
                    else:
                        self.direction = LEFT
                        return DOWN
                elif self.direction == LEFT:
                    if currentCell[1] > 1:
                        return LEFT
                    else:
                        self.direction = RIGHT
                        return DOWN
            else:
                return DOWN
        else:
            return DOWN