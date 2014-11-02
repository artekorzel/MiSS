typedef struct DropletParameter {
    float cutoffRadius;
    float mass;
    float repulsionParameter;
    float lambda;
    float sigma;
    float gamma;
} DropletParameter;

float weightR(float distanceValue, float cutoffRadius) {
    if(distanceValue > cutoffRadius) {
        return 0.0;
    }
    return 1.0 - distanceValue / cutoffRadius;
}

float3 normalizePosition(float3 vector, float boxSize) {
    return fmod(fmod(vector + boxSize, 2.0f * boxSize) + 2.0f * boxSize, 2.0f * boxSize) - boxSize;
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

// <0; 1>
float rand(int* seed, int step) {
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = (*seed * a * step) % m;
    return (float)(*seed) / m;
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
    float U1 = (rand(&seed, step) + 1.0) / 2;
    float U2 = (rand(&seed, step) + 1.0) / 2;
    return normalRand(U1, U2);
}

int calculateCellId(float3 position, float cellRadius, float boxSize) {
    return ((int)((position.x + boxSize) / cellRadius)) + 
            ((int)(2 * boxSize / cellRadius)) * (((int)((position.y + boxSize) / cellRadius)) + 
                    ((int)(2 * boxSize / cellRadius)) * ((int)((position.z + boxSize) / cellRadius)));
}

float3 calculateForce(global float3* positions, global float3* velocities, global DropletParameter* params,
        global int* types, global int* cells, global int* cellNeighbours, float cellRadius, float boxSize,
        int maxDropletsPerCell, int numberOfCells, int dropletId, int dropletCellNeighbourId, int step) {

    float3 conservativeForce = (float3)(0.0, 0.0, 0.0);
    float3 dissipativeForce = (float3)(0.0, 0.0, 0.0);
    float3 randomForce = (float3)(0.0, 0.0, 0.0);

    float3 dropletPosition = positions[dropletId];
    float3 dropletVelocity = velocities[dropletId];
    
    int dropletType = types[dropletId];
    DropletParameter dropletParameter = params[dropletType];
    
    float repulsionParameter = dropletParameter.repulsionParameter;
    float gamma = dropletParameter.gamma;
    float sigma = dropletParameter.sigma;
    
    int dropletCellId = calculateCellId(dropletPosition, cellRadius, boxSize);
    global int* dropletCellNeighbours = &cellNeighbours[dropletCellId * 27];
    
    int i, j, neighbourId;
    int cellId = dropletCellNeighbours[dropletCellNeighbourId];
    if(cellId >= 0) {
        global int* dropletNeighbours = &cells[maxDropletsPerCell * cellId];
        for(j = 0, neighbourId = dropletNeighbours[j]; neighbourId >= 0; neighbourId = dropletNeighbours[++j]) {
            if(neighbourId != dropletId) {
                float3 neighbourPosition = positions[neighbourId];
                float distanceValue = distance(neighbourPosition, dropletPosition);

                int neighbourType = types[neighbourId];
                DropletParameter neighbourParameter = params[neighbourType];
                float cutoffRadius = neighbourParameter.cutoffRadius;

                if(distanceValue < cutoffRadius) {
                    float3 normalizedPositionVector = normalize(neighbourPosition - dropletPosition);
                    if(dropletType != 0 || neighbourType != 0) {
                        float weightRValue = weightR(distanceValue, cutoffRadius);
                        float weightDValue = weightRValue * weightRValue;

                        conservativeForce -= sqrt(repulsionParameter * neighbourParameter.repulsionParameter)
                                * (1.0f - distanceValue / cutoffRadius) * normalizedPositionVector;

                        dissipativeForce += gamma * weightDValue * normalizedPositionVector
                                * dot(normalizedPositionVector, velocities[neighbourId] - dropletVelocity);

                        randomForce -= sigma * weightRValue * normalizedPositionVector
                                * gaussianRandom(dropletId, neighbourId, step);
                    }
                }
            }
        }
    }
    
    if(dropletType == 0 || step > 100) {
        return conservativeForce + dissipativeForce + randomForce;
    } else {
        return conservativeForce + dissipativeForce + randomForce + (float3)(0.0f, 0.01f, 0.0f);    
    }
}

kernel void fillCells(global int* cells, global float3* positions, float cellRadius, 
        float boxSize, int numberOfDroplets, int maxDropletsPerCell, int numberOfCells) {

    int cellId = get_global_id(0);
    if (cellId >= numberOfCells) {
        return;
    }
    
    int dropletId = 0, freeId = 0;
    for(; dropletId < numberOfDroplets; ++dropletId) {
        float3 position = positions[dropletId];
        int predictedCellId = calculateCellId(position, cellRadius, boxSize);
        if(predictedCellId == cellId) {
            cells[maxDropletsPerCell * cellId + freeId++] = dropletId;
        }
    }
    for(; freeId < maxDropletsPerCell; ++freeId) {
        cells[maxDropletsPerCell * cellId + freeId] = -1;
    }
}

kernel void fillCellNeighbours(global int* cellNeighbours, 
        float cellRadius, float boxSize, int numberOfCells) {
    
    int cellId = get_global_id(0);
    if (cellId >= numberOfCells) {
        return;
    }
    
    int numberOfCellsPerDim = ceil(2 * boxSize / cellRadius);
    int squareOfNumberOfCellsPerDim = numberOfCellsPerDim * numberOfCellsPerDim;
    
    int cellIdPartX = cellId % numberOfCellsPerDim;
    int cellIdPartY = (cellId / numberOfCellsPerDim) % numberOfCellsPerDim;
    int cellIdPartZ = cellId / squareOfNumberOfCellsPerDim;

    int cellIndex = cellId * 27;
    cellNeighbours[cellIndex++] = cellId;
    
    if(cellIdPartX > 0) {
        cellNeighbours[cellIndex++] = cellId - 1;
    }
    
    if(cellIdPartX < numberOfCellsPerDim - 1) {
        cellNeighbours[cellIndex++] = cellId + 1;
    }
    
    if(cellIdPartY > 0) {
        cellNeighbours[cellIndex++] = cellId - numberOfCellsPerDim;
        
        if(cellIdPartX > 0) {
            cellNeighbours[cellIndex++] = cellId - numberOfCellsPerDim - 1;
        }

        if(cellIdPartX < numberOfCellsPerDim - 1) {
            cellNeighbours[cellIndex++] = cellId - numberOfCellsPerDim + 1;
        }
    }
    
    if(cellIdPartY < numberOfCellsPerDim - 1) {
        cellNeighbours[cellIndex++] = cellId + numberOfCellsPerDim;
        
        if(cellIdPartX > 0) {
            cellNeighbours[cellIndex++] = cellId + numberOfCellsPerDim - 1;
        }

        if(cellIdPartX < numberOfCellsPerDim - 1) {
            cellNeighbours[cellIndex++] = cellId + numberOfCellsPerDim + 1;
        }
    }
    
    if(cellIdPartZ > 0) {
        cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim;
        
        if(cellIdPartX > 0) {
            cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - 1;
        }

        if(cellIdPartX < numberOfCellsPerDim - 1) {
            cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + 1;
        }

        if(cellIdPartY > 0) {
            cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - numberOfCellsPerDim;

            if(cellIdPartX > 0) {
                cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - numberOfCellsPerDim - 1;
            }

            if(cellIdPartX < numberOfCellsPerDim - 1) {
                cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - numberOfCellsPerDim + 1;
            }
        }

        if(cellIdPartY < numberOfCellsPerDim - 1) {
            cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + numberOfCellsPerDim;

            if(cellIdPartX > 0) {
                cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + numberOfCellsPerDim - 1;
            }

            if(cellIdPartX < numberOfCellsPerDim - 1) {
                cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + numberOfCellsPerDim + 1;
            }
        }
    }
    
    if(cellIdPartZ < numberOfCellsPerDim - 1) {
        cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim;
        
        if(cellIdPartX > 0) {
            cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - 1;
        }

        if(cellIdPartX < numberOfCellsPerDim - 1) {
            cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + 1;
        }

        if(cellIdPartY > 0) {
            cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - numberOfCellsPerDim;

            if(cellIdPartX > 0) {
                cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - numberOfCellsPerDim - 1;
            }

            if(cellIdPartX < numberOfCellsPerDim - 1) {
                cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - numberOfCellsPerDim + 1;
            }
        }

        if(cellIdPartY < numberOfCellsPerDim - 1) {
            cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + numberOfCellsPerDim;

            if(cellIdPartX > 0) {
                cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + numberOfCellsPerDim - 1;
            }

            if(cellIdPartX < numberOfCellsPerDim - 1) {
                cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + numberOfCellsPerDim + 1;
            }
        }
    }
    
    for(int n = (cellId + 1) * 27; cellIndex < n; ++cellIndex) {
        cellNeighbours[cellIndex] = -1;
    }
}

kernel void calculateForces(global float3* positions, global float3* velocities, global float3* forces, 
        global DropletParameter* params, global int* types, global int* cells, global int* cellNeighbours, 
        float cellRadius, float boxSize,int numberOfDroplets, int maxDropletsPerCell, 
        int numberOfCells, int step) {

    int dropletId = get_global_id(0) / get_local_size(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    int dropletCellNeighbourId = get_local_id(0);
    if(dropletCellNeighbourId >= 27) {
        return;
    }

    local float3 localForces[27];
    localForces[dropletCellNeighbourId] = calculateForce(positions, velocities, params, types, 
            cells, cellNeighbours, cellRadius, boxSize, maxDropletsPerCell, numberOfCells, 
            dropletId, dropletCellNeighbourId, step);
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    if(dropletCellNeighbourId == 0) {
        float3 force = localForces[0];
        for(int i = 1; i < 27; ++i) {
            force += localForces[i];
        }
        forces[dropletId] = force;
    }
}

kernel void calculateNewPositionsAndPredictedVelocities(global float3* positions, global float3* velocities,
        global float3* forces, global float3* newPositions, global float3* predictedVelocities,
        global DropletParameter* params, global int* types, float deltaTime, int numberOfDroplets, 
        float boxSize) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    DropletParameter dropletParameter = params[types[dropletId]];
    float3 dropletVelocity = velocities[dropletId];
    float3 dropletForce = forces[dropletId];
    float dropletMass = dropletParameter.mass;

    float3 newPosition = positions[dropletId] + deltaTime * dropletVelocity
            + 0.5f * deltaTime * deltaTime * dropletForce / dropletMass;
            
    newPositions[dropletId] = normalizePosition(newPosition, boxSize);
    
    predictedVelocities[dropletId] = dropletVelocity 
            + dropletParameter.lambda * deltaTime * dropletForce / dropletMass;
}

kernel void calculateNewVelocities(global float3* newPositions, global float3* velocities,
        global float3* predictedVelocities, global float3* newVelocities, global float3* forces,
        global DropletParameter* params, global int* types, global int* cells, global int* cellNeighbours, 
        float deltaTime, float cellRadius, float boxSize, int numberOfDroplets, int maxDropletsPerCell,
        int numberOfCells, int step) {

    int dropletId = get_global_id(0) / 27;
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    int dropletCellNeighbourId = get_local_id(0);
    if(dropletCellNeighbourId >= 27) {
        return;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    local float3 localForces[27];
    localForces[dropletCellNeighbourId] = calculateForce(newPositions, predictedVelocities, 
            params, types, cells, cellNeighbours, cellRadius, boxSize, maxDropletsPerCell, 
            numberOfCells, dropletId, dropletCellNeighbourId, step);
    
    if(dropletCellNeighbourId == 0) {
        float3 predictedForce = localForces[0];
        for(int i = 1; i < 27; ++i) {
            predictedForce += localForces[i];
        }
        newVelocities[dropletId] = velocities[dropletId] + 0.5f * deltaTime 
                * (forces[dropletId] + predictedForce) / params[types[dropletId]].mass;
    }
}

kernel void generateTube(global float3* vector, global int* types, global int* states, 
        int numberOfDroplets, float radiusIn, float radiusOut, float height) {
    
    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    int seed = states[dropletId];   
    
    float x = (rand(&seed, 1) * 2 - 1) * radiusOut;
    float y = (rand(&seed, 1) * 2 - 1) * height;
    float rangeOut = sqrt(radiusOut * radiusOut - x * x);
    float z = (rand(&seed, 1) * 2 - 1) * rangeOut;
    
    float distanceFromY = sqrt(x * x + z * z);
    if (distanceFromY >= radiusIn) {
        types[dropletId] = 0;
    } else {
        float randomNum = rand(&seed, 1);
        if (randomNum >= 0.5f) {
            types[dropletId] = 1;    
        } else {
            types[dropletId] = 2;
        }        
    }
        
    states[dropletId] = seed;
    vector[dropletId] = (float3) (x, y, z);
}

kernel void generateRandomVector(global float3* vector, global int* states, global int* types, 
        float thermalVelocity, float flowVelocity, int numberOfDroplets) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    float x, y, z;
    int seed = states[dropletId];
    if(types[dropletId] == 0){
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    } else {
        x = (rand(&seed, 1) * 2 - 1) * thermalVelocity;
        y = ((rand(&seed, 1) * 2 - 1) * flowVelocity + flowVelocity) / 2.0f;
        z = (rand(&seed, 1) * 2 - 1) * thermalVelocity;
    }
    states[dropletId] = seed;
    vector[dropletId] = ((float3) (x, y, z));
}

kernel void doVectorReduction(global float3* data, global float3* partialSums, 
        global float3* output, int dataLength) {

    int global_id = get_global_id(0);
    int group_size = get_global_size(0);

    if(global_id < dataLength) {
        partialSums[global_id] = data[global_id];
    } else {
        partialSums[global_id] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    int offset;
    for(offset = group_size/2; offset > 0; offset >>= 1) {
        if(global_id < offset){
            partialSums[global_id] += partialSums[global_id + offset];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    
    if(global_id == 0) {
        output[0] = partialSums[0]/dataLength;
    }
}
