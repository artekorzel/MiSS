int cellIdConditionZeroX(int cellIdPartX, int baseCellId, int numberOfCellsPerXZDim) {
    if(cellIdPartX == 0) {
        return baseCellId + numberOfCellsPerXZDim;
    }
    return baseCellId;
}

int cellIdConditionLastX(int cellIdPartX, int baseCellId, int numberOfCellsPerXZDim) {
    if(cellIdPartX == numberOfCellsPerXZDim - 1) {
        return baseCellId - numberOfCellsPerXZDim;
    }
    return baseCellId;
}

int cellIdConditionZeroY(int cellIdPartY, int baseCellId, 
        int squareOfNumberOfCellsPerDim, int numberOfCellsPerYDim) {
    if(cellIdPartY == 0) {
        return baseCellId + squareOfNumberOfCellsPerDim;
    }
    return baseCellId;
}

int cellIdConditionLastY(int cellIdPartY, int baseCellId, 
        int squareOfNumberOfCellsPerDim, int numberOfCellsPerYDim) {
    if(cellIdPartY == numberOfCellsPerYDim - 1) {
        return baseCellId - squareOfNumberOfCellsPerDim;
    }
    return baseCellId;
}

int cellIdConditionZeroZ(int cellIdPartZ, int baseCellId, 
        int numberOfCellsPerXZDim, int numberOfCells) {
    if(cellIdPartZ == 0) {
        return baseCellId + numberOfCells;
    }
    return baseCellId;
}

int cellIdConditionLastZ(int cellIdPartZ, int baseCellId, 
        int numberOfCellsPerXZDim, int numberOfCells) {
    if(cellIdPartZ == numberOfCellsPerXZDim - 1) {
        return baseCellId - numberOfCells;
    }
    return baseCellId;
}

kernel void fillCellNeighbours(global int* cellNeighbours, 
        float cellRadius, float boxSize, float boxWidth, int numberOfCells) {
    
    int cellId = get_global_id(0);
    if (cellId >= numberOfCells) {
        return;
    }
    
    int numberOfCellsPerXZDim = ceil(2 * boxSize / cellRadius);
    int numberOfCellsPerYDim = ceil(2 * boxWidth / cellRadius);
    int squareOfNumberOfCellsPerDim = numberOfCellsPerXZDim * numberOfCellsPerYDim;
    
    int cellIdPartX = cellId % numberOfCellsPerXZDim;
    int cellIdPartY = (cellId / numberOfCellsPerXZDim) % numberOfCellsPerYDim;
    int cellIdPartZ = cellId / squareOfNumberOfCellsPerDim;

    int cellIndex = cellId * 27;
    cellNeighbours[cellIndex++] = cellId;    
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellId - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellId + 1, numberOfCellsPerXZDim);

    int cellIdConditionYValue = cellIdConditionZeroY(cellIdPartY, cellId - numberOfCellsPerXZDim, squareOfNumberOfCellsPerDim, numberOfCellsPerYDim);
    cellNeighbours[cellIndex++] = cellIdConditionYValue;
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionYValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionYValue + 1, numberOfCellsPerXZDim);

    cellIdConditionYValue = cellIdConditionLastY(cellIdPartY, cellId + numberOfCellsPerXZDim, squareOfNumberOfCellsPerDim, numberOfCellsPerYDim);
    cellNeighbours[cellIndex++] = cellIdConditionYValue;
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionYValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionYValue + 1, numberOfCellsPerXZDim);

    int cellIdConditionZValue = cellIdConditionZeroZ(cellIdPartZ, cellId - squareOfNumberOfCellsPerDim, numberOfCellsPerXZDim, numberOfCells);
    cellNeighbours[cellIndex++] = cellIdConditionZValue;    
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionZValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionZValue + 1, numberOfCellsPerXZDim);

    cellIdConditionYValue = cellIdConditionZeroY(cellIdPartY, cellIdConditionZValue - numberOfCellsPerXZDim, squareOfNumberOfCellsPerDim, numberOfCellsPerYDim);
    cellNeighbours[cellIndex++] = cellIdConditionYValue;
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionYValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionYValue + 1, numberOfCellsPerXZDim);

    cellIdConditionYValue = cellIdConditionLastY(cellIdPartY, cellIdConditionZValue + numberOfCellsPerXZDim, squareOfNumberOfCellsPerDim, numberOfCellsPerYDim);
    cellNeighbours[cellIndex++] = cellIdConditionYValue;
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionYValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionYValue + 1, numberOfCellsPerXZDim);

    cellIdConditionZValue = cellIdConditionLastZ(cellIdPartZ, cellId + squareOfNumberOfCellsPerDim, numberOfCellsPerXZDim, numberOfCells);
    cellNeighbours[cellIndex++] = cellIdConditionZValue;    
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionZValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionZValue + 1, numberOfCellsPerXZDim);

    cellIdConditionYValue = cellIdConditionZeroY(cellIdPartY, cellIdConditionZValue - numberOfCellsPerXZDim, squareOfNumberOfCellsPerDim, numberOfCellsPerYDim);
    cellNeighbours[cellIndex++] = cellIdConditionYValue;
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionYValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionYValue + 1, numberOfCellsPerXZDim);

    cellIdConditionYValue = cellIdConditionLastY(cellIdPartY, cellIdConditionZValue + numberOfCellsPerXZDim, squareOfNumberOfCellsPerDim, numberOfCellsPerYDim);
    cellNeighbours[cellIndex++] = cellIdConditionYValue;
    cellNeighbours[cellIndex++] = cellIdConditionZeroX(cellIdPartX, cellIdConditionYValue - 1, numberOfCellsPerXZDim);
    cellNeighbours[cellIndex++] = cellIdConditionLastX(cellIdPartX, cellIdConditionYValue + 1, numberOfCellsPerXZDim);
}