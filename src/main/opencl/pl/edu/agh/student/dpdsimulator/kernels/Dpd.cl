typedef struct Parameters {
    float mass;
    float lambda;
    float cutoffRadius;
    float pi;
    float sigma;
    float gamma;
} Parameters;

float weightR(float distanceValue, float cutoffRadius) {
    if(distanceValue > cutoffRadius) {
        return 0.0;
    }
    return 1.0 - distanceValue / cutoffRadius;
}

float3 normalizePosition(float3 vector, float boxSize, float boxWidth) {
    float3 changeVector = (float3)(boxSize, boxWidth, boxSize);
    return fmod(fmod(vector + changeVector, 2.0f * changeVector) + 2.0f * changeVector, 2.0f * changeVector) - changeVector;
}

int calculateHash(int d1, int d2) {    
    int i1, i2;
    if(d1 <= d2) {
        i1 = d1;
        i2 = d2;
    } else {
        i1 = d2;
        i2 = d1;
    }
    return (i1 + i2) * (i1 + i2 + 1) / 2 + i2;
}

// <0; 1)
float rand(int* seed, int step) {
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = (*seed * a * step) % m;
    float randomValue = (float)(*seed) / m;
    if(randomValue < 0) {
        return -randomValue;
    }
    return randomValue;
}

// <-1; 1>
float normalRand(float U1, float U2) {
     float R = -2 * log(U1);
     float fi = 2 * M_PI * U2;
     float Z1 = sqrt(R) * cos(fi);
     return Z1;
     //float Z2 = sqrt(R) * sin(fi);
}

// <-1; 1>
float gaussianRandom(int dropletId, int neighbourId, int step) {
    int seed = calculateHash(dropletId, neighbourId);
    float U1 = rand(&seed, step);
    float U2 = rand(&seed, step);
    return normalRand(U1, U2);
}

int calculateCellId(float3 position, float cellRadius, float boxSize, float boxWidth) {
    return ((int)((position.x + boxSize) / cellRadius)) + 
            ((int)(2 * boxSize / cellRadius)) * (((int)((position.y + boxWidth) / cellRadius)) + 
                    ((int)(2 * boxWidth / cellRadius)) * ((int)((position.z + boxSize) / cellRadius)));
}

int3 calculateCellCoordinates(int dropletCellId, float cellRadius, float boxSize, 
        float boxWidth, int cellsNoXZ, int cellsNoY) {
    int dropletCellZ = dropletCellId / (cellsNoXZ * cellsNoY);
    int dropletCellX = dropletCellId - dropletCellZ * cellsNoXZ * cellsNoY;
    int dropletCellY = dropletCellX / cellsNoXZ;
    dropletCellX = dropletCellId % cellsNoXZ;
    return (int3)(dropletCellX, dropletCellY, dropletCellZ);
}

float3 getNeighbourPosition(global float3* positions, int3 dropletCellCoordinates, 
        float cellRadius, int dropletCellId, int neighbourCellId, int neighbourId, 
        float boxSize, float boxWidth, int cellsNoXZ, int cellsNoY) {
    float3 position = positions[neighbourId];
    
    if(neighbourCellId != dropletCellId) {
        int3 neighbourCellCoordinates = calculateCellCoordinates(neighbourCellId, cellRadius, boxSize, boxWidth, cellsNoXZ, cellsNoY);

        if(dropletCellCoordinates.x == 0 && neighbourCellCoordinates.x == cellsNoXZ - 1) {
            position.x -= boxSize;
        }
        
        if(dropletCellCoordinates.x == cellsNoXZ - 1 && neighbourCellCoordinates.x == 0) {
            position.x += boxSize;
        }
        
        if(dropletCellCoordinates.y == 0 && neighbourCellCoordinates.y == cellsNoY - 1) {
            position.y -= boxWidth;
        }
        
        if(dropletCellCoordinates.y == cellsNoY - 1 && neighbourCellCoordinates.y == 0) {
            position.y += boxWidth;
        }
        
        if(dropletCellCoordinates.z == 0 && neighbourCellCoordinates.z == cellsNoXZ - 1) {
            position.z -= boxSize;
        }
        
        if(dropletCellCoordinates.z == cellsNoXZ - 1 && neighbourCellCoordinates.z == 0) {
            position.z += boxSize;
        }
    }
    
    return position;
}

float3 calculateForce(global float3* positions, global float3* velocities, global Parameters* params,
        global int* cells, global int* cellNeighbours, float cellRadius, float boxSize, 
        float boxWidth, int maxDropletsPerCell, int numberOfCells, int dropletId, int step) {

    float3 conservativeForce = (float3)(0.0, 0.0, 0.0);
    float3 dissipativeForce = (float3)(0.0, 0.0, 0.0);
    float3 randomForce = (float3)(0.0, 0.0, 0.0);
    
    Parameters parameters = params[0];
    float pi = parameters.pi;
    float gamma = parameters.gamma;
    float sigma = parameters.sigma;
    float cutoffRadius = parameters.cutoffRadius;

    float3 dropletPosition = positions[dropletId];
    float3 dropletVelocity = velocities[dropletId];    
    int dropletCellId = calculateCellId(dropletPosition, cellRadius, boxSize, boxWidth);
    int cellsNoXZ = (int)(2 * boxSize / cellRadius);
    int cellsNoY = (int)(2 * boxWidth / cellRadius);
    int3 dropletCellCoordinates = calculateCellCoordinates(dropletCellId, cellRadius, boxSize, boxWidth, cellsNoXZ, cellsNoY);
    
    int j, neighbourId, dropletCellNeighbourId;
    for(dropletCellNeighbourId = 0; dropletCellNeighbourId < 27; ++dropletCellNeighbourId) {
        int cellId = cellNeighbours[dropletCellId * 27 + dropletCellNeighbourId];
        global int* dropletNeighbours = &cells[maxDropletsPerCell * cellId];
        for(j = 0, neighbourId = dropletNeighbours[j]; neighbourId >= 0; neighbourId = dropletNeighbours[++j]) {
            if(neighbourId != dropletId) {
                float3 neighbourPosition = getNeighbourPosition(positions, dropletCellCoordinates, 
                        cellRadius, dropletCellId, cellId, neighbourId, boxSize, boxWidth, cellsNoXZ, cellsNoY);
                float distanceValue = distance(neighbourPosition, dropletPosition);

                if(distanceValue < cutoffRadius) {
                    float3 normalizedPositionVector = normalize(neighbourPosition - dropletPosition);
                    float weightRValue = weightR(distanceValue, cutoffRadius);
                    float weightDValue = weightRValue * weightRValue;
                    
                    conservativeForce += pi * (1.0f - distanceValue / cutoffRadius) * normalizedPositionVector;
                    
                    dissipativeForce -= gamma * weightDValue * normalizedPositionVector
                            * dot(velocities[neighbourId] - dropletVelocity, normalizedPositionVector);

                    randomForce += sigma * weightRValue * normalizedPositionVector
                            * gaussianRandom(dropletId, neighbourId, step);
                }
            }
        }
    }
    
    return conservativeForce + dissipativeForce + randomForce;
}

kernel void calculateForces(global float3* positions, global float3* velocities, global float3* forces, 
        global Parameters* params, global int* cells, global int* cellNeighbours, float cellRadius, float boxSize, 
        float boxWidth, int numberOfDroplets, int maxDropletsPerCell, int numberOfCells, int step) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    forces[dropletId] = calculateForce(positions, velocities, params, cells, cellNeighbours, 
            cellRadius, boxSize, boxWidth, maxDropletsPerCell, numberOfCells, dropletId, step);
}

kernel void calculateNewPositionsAndVelocities(global float3* positions, global float3* velocities,
        global float3* forces, global float3* newPositions, global float3* newVelocities,
        global Parameters* params, float deltaTime, int numberOfDroplets, float boxSize, float boxWidth) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    float3 dropletVelocity = velocities[dropletId];
    float3 dropletForce = forces[dropletId];
    float dropletMass = params[0].mass;

    float3 newPosition = positions[dropletId] + deltaTime * dropletVelocity;            
    newPositions[dropletId] = normalizePosition(newPosition, boxSize, boxWidth);
    
    newVelocities[dropletId] = velocities[dropletId] + deltaTime * forces[dropletId] / params[0].mass;
}

kernel void generateTube(global float3* vector, global int* states, 
        int numberOfDroplets, float boxSize, float boxWidth) {
    
    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    int seed = states[dropletId];   
    
    float x = (rand(&seed, 1) * 2 - 1) * boxSize;
    float y = (rand(&seed, 1) * 2 - 1) * boxWidth;
    float z = (rand(&seed, 1) * 2 - 1) * boxSize;
        
    states[dropletId] = seed;
    vector[dropletId] = (float3) (x, y, z);
}

kernel void generateRandomVector(global float3* vector, global int* states, int numberOfDroplets) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    vector[dropletId] = ((float3) (0, 0, 0));
}

kernel void calculateVelocitiesEnergy(global float3* velocities, global float* energy, int numberOfDroplets) {

    int globalId = get_global_id(0);
    if (globalId >= numberOfDroplets) {
        return;
    }    

    float3 velocity = velocities[globalId];
    energy[globalId] = velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
}

kernel void fillCells(global int* cells, global float3* positions, float cellRadius, 
        float boxSize, float boxWidth, int numberOfDroplets, int maxDropletsPerCell, int numberOfCells) {

    int cellId = get_global_id(0);
    if (cellId >= numberOfCells) {
        return;
    }
    
    int dropletId = 0, freeId = 0;
    for(; dropletId < numberOfDroplets; ++dropletId) {
        float3 position = positions[dropletId];
        int predictedCellId = calculateCellId(position, cellRadius, boxSize, boxWidth);
        if(predictedCellId == cellId) {
            cells[maxDropletsPerCell * cellId + freeId++] = dropletId;
        }
    }
    for(; freeId < maxDropletsPerCell; ++freeId) {
        cells[maxDropletsPerCell * cellId + freeId] = -1;
    }
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
    
    int neighboursX[3];
    int neighboursY[3];
    int neighboursZ[3];

    neighboursX[0] = cellIdPartX;
    neighboursX[1] = (cellIdPartX + numberOfCellsPerXZDim - 1) % numberOfCellsPerXZDim;
    neighboursX[2] = (cellIdPartX + numberOfCellsPerXZDim + 1) % numberOfCellsPerXZDim;
    
    neighboursY[0] = cellIdPartY;
    neighboursY[1] = (cellIdPartY + numberOfCellsPerYDim - 1) % numberOfCellsPerYDim;
    neighboursY[2] = (cellIdPartY + numberOfCellsPerYDim + 1) % numberOfCellsPerYDim;

    neighboursZ[0] = cellIdPartZ;
    neighboursZ[1] = (cellIdPartZ + numberOfCellsPerXZDim - 1) % numberOfCellsPerXZDim;
    neighboursZ[2] = (cellIdPartZ + numberOfCellsPerXZDim + 1) % numberOfCellsPerXZDim;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 3; k++){
                cellNeighbours[cellIndex++] = neighboursX[i] + neighboursY[j] * numberOfCellsPerXZDim + neighboursZ[k] * squareOfNumberOfCellsPerDim;
            }
        }
    }
}
