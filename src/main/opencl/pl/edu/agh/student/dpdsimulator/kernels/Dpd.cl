typedef struct SimulationParameters {
    float boxSizeX;
    float boxSizeY;
    float boxSizeZ;
    int cellsNoX;
    int cellsNoY;
    int cellsNoZ;
    float cellRadius;
    int maxDropletsPerCell;
    int numberOfDroplets;
    int numberOfCells;
    int numberOfTypes;
    float deltaTime;
    float radiusIn;
    float accelerationVesselPart;
    float accelerationValue;
    int accelerationVeselSteps;
    float averageDropletDistance;
    bool shouldSimulateVesselDroplets;
} SimulationParameters;

typedef struct DropletParameters {
    float mass;
} DropletParameters;

typedef struct PairParameters {
    float cutoffRadius;
    float pi;
    float sigma;
    float gamma;
} PairParameters;

float weightR(float distanceValue, float cutoffRadius) {
    return 1.0f / distanceValue - 1.0f / cutoffRadius;
}

float3 normalizePosition(float3 vector, float boxSizeX, float boxSizeY, float boxSizeZ) {
    float3 changeVector = (float3)(boxSizeX, boxSizeY, boxSizeZ);
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
float pairRandom(int dropletId, int neighbourId, int step) {
    int seed = calculateHash(dropletId, neighbourId);
    return 2 * rand(&seed, step) - 1;
}

int calculateCellId(float3 position, float boxSizeX, float boxSizeY, float boxSizeZ) {
    return ((int)((position.x + boxSizeX))) + 
            ((int)(2 * boxSizeX)) * (((int)((position.y + boxSizeY))) + 
                    ((int)(2 * boxSizeY)) * ((int)((position.z + boxSizeZ))));
}

int3 calculateCellCoordinates(int dropletCellId, float boxSizeX, float boxSizeY, 
        float boxSizeZ, int cellsNoX, int cellsNoY, int cellsNoZ) {
    int dropletCellZ = dropletCellId / (cellsNoX * cellsNoY);
    int dropletCellX = dropletCellId - dropletCellZ * cellsNoX * cellsNoY;
    int dropletCellY = dropletCellX / cellsNoX;
    dropletCellX = dropletCellId % cellsNoX;
    return (int3)(dropletCellX, dropletCellY, dropletCellZ);
}

float3 getNeighbourPosition(global float3* positions, int3 dropletCellCoordinates, 
        int dropletCellId, int neighbourCellId, int neighbourId, 
        float boxSizeX, float boxSizeY, float boxSizeZ, int cellsNoX, int cellsNoY, int cellsNoZ) {
    float3 position = positions[neighbourId];
    
    if(neighbourCellId != dropletCellId) {
        int3 neighbourCellCoordinates = calculateCellCoordinates(neighbourCellId, 
                boxSizeX, boxSizeY, boxSizeZ, cellsNoX, cellsNoY, cellsNoZ);
                
        if(dropletCellCoordinates.x == 0 && neighbourCellCoordinates.x == cellsNoX - 1) {
            position.x -= 2 * boxSizeX;
        }
        
        if(dropletCellCoordinates.x == cellsNoX - 1 && neighbourCellCoordinates.x == 0) {
            position.x += 2 * boxSizeX;
        }
        
        if(dropletCellCoordinates.y == 0 && neighbourCellCoordinates.y == cellsNoY - 1) {
            position.y -= 2 * boxSizeY;
        }
        
        if(dropletCellCoordinates.y == cellsNoY - 1 && neighbourCellCoordinates.y == 0) {
            position.y += 2 * boxSizeY;
        }
        
        if(dropletCellCoordinates.z == 0 && neighbourCellCoordinates.z == cellsNoZ - 1) {
            position.z -= 2 * boxSizeZ;
        }
        
        if(dropletCellCoordinates.z == cellsNoZ - 1 && neighbourCellCoordinates.z == 0) {
            position.z += 2 * boxSizeZ;
        }
    }
    
    return position;
}

float3 calculateForce(global float3* positions, global float3* velocities, global float3* velocities0, global int* types, 
        global int* cells, global int* cellNeighbours, constant PairParameters* pairParams, constant DropletParameters* dropletParams, 
        SimulationParameters simulationParams, int dropletId, int step, global float* forces0) {
            
    int dropletType = types[dropletId];
        
    if(!simulationParams.shouldSimulateVesselDroplets
            && dropletType == 0) {
        return (float3)(0.0, 0.0, 0.0);
    }
    
    float boxSizeX = simulationParams.boxSizeX;
    float boxSizeY = simulationParams.boxSizeY;
    float boxSizeZ = simulationParams.boxSizeZ;
    int cellsNoX = simulationParams.cellsNoX;
    int cellsNoY = simulationParams.cellsNoY;
    int cellsNoZ = simulationParams.cellsNoZ;
    int maxDropletsPerCell = simulationParams.maxDropletsPerCell;
    int numberOfTypes = simulationParams.numberOfTypes;
        
    float3 conservativeForce = (float3)(0.0, 0.0, 0.0);
    float3 dissipativeForce = (float3)(0.0, 0.0, 0.0);
    float3 randomForce = (float3)(0.0, 0.0, 0.0);
    
    float3 dropletPosition = positions[dropletId];
    float3 dropletVelocity = velocities0[dropletId];    
    int dropletCellId = calculateCellId(dropletPosition, boxSizeX, boxSizeY, boxSizeZ);
    int3 dropletCellCoordinates = calculateCellCoordinates(dropletCellId,
            boxSizeX, boxSizeY, boxSizeZ, cellsNoX, cellsNoY, cellsNoZ);
    
    int j, neighbourId, dropletCellNeighbourId, noOfNeighbours = 0;
    for(dropletCellNeighbourId = 0; dropletCellNeighbourId < 27; ++dropletCellNeighbourId) {
        int cellId = cellNeighbours[dropletCellId * 27 + dropletCellNeighbourId];
        global int* dropletNeighbours = &cells[maxDropletsPerCell * cellId];
        for(j = 0, neighbourId = dropletNeighbours[j]; neighbourId >= 0; neighbourId = dropletNeighbours[++j]) {
            if(neighbourId != dropletId) {
                int neighbourType = types[neighbourId];
                
                float3 neighbourPosition = getNeighbourPosition(positions, dropletCellCoordinates,
                        dropletCellId, cellId, neighbourId, boxSizeX, boxSizeY, boxSizeZ, cellsNoX, cellsNoY, cellsNoZ);
                float distanceValue = distance(neighbourPosition, dropletPosition);

                PairParameters pairParameters = pairParams[dropletType * numberOfTypes + neighbourType];
                float cutoffRadius = pairParameters.cutoffRadius;

                if(distanceValue < cutoffRadius) {
                    float3 normalizedPositionVector = neighbourPosition - dropletPosition;
                    float weightRValue = weightR(distanceValue, cutoffRadius);
                    float weightDValue = weightRValue * weightRValue;

                    float pi = pairParameters.pi;
                    float gamma = pairParameters.gamma;
                    float sigma = pairParameters.sigma;

                    conservativeForce += pi * weightRValue * normalizedPositionVector;

                    dissipativeForce -= gamma * weightDValue * normalizedPositionVector
                            * dot(velocities0[neighbourId] - dropletVelocity, normalizedPositionVector);

                    
                    if(dropletCellId == cellId) {
                    randomForce += sigma * weightRValue * normalizedPositionVector;
//                            * pairRandom(dropletId, neighbourId, step);
                    } else {
                        randomForce -= sigma * weightRValue * normalizedPositionVector;
                    }
                            
                    ++noOfNeighbours;
                }
            }
        }
    }
    
    float3 force;
    if(dropletType != 0 
            && dropletPosition.y < simulationParams.accelerationVesselPart 
            && step < simulationParams.accelerationVeselSteps) {
        force = conservativeForce + dissipativeForce + randomForce + (float3)(0, simulationParams.accelerationValue, 0);
    } else {
        force = conservativeForce + dissipativeForce + randomForce;
    }
    
    return force / (noOfNeighbours + 1);
}

kernel void calculateForces(global float3* positions, global float3* velocities, global float3* velocities0, global float3* forces, 
        global float* forces0, global int* types, global int* cells, global int* cellNeighbours, constant PairParameters* pairParams, 
        constant DropletParameters* dropletParams, constant SimulationParameters* simulationParameters, int step) {

    SimulationParameters simulationParams = simulationParameters[0];

    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    forces[dropletId] = calculateForce(positions, velocities, velocities0, types, cells, 
            cellNeighbours, pairParams, dropletParams, simulationParams, dropletId, step, forces0);
}

kernel void calculateNewPositionsAndVelocities(global float3* positions, global float3* velocities,
        global float3* velocities0, global float3* forces, global float* dropletsEnergy, global int* types, 
        constant PairParameters* pairParams, constant DropletParameters* dropletParams, 
        constant SimulationParameters* simulationParameters) {

    SimulationParameters simulationParams = simulationParameters[0];
    
    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    if(!simulationParams.shouldSimulateVesselDroplets 
            && types[dropletId] == 0) {
        return;
    }
        
    float3 dropletForce = forces[dropletId];
    float3 dropletVelocity = velocities[dropletId] - dropletForce;
    float3 dropletVelocityBuf = 0.5f * (velocities[dropletId] + dropletVelocity);
    float dropletMass = dropletParams[types[dropletId]].mass;
    
    dropletsEnergy[dropletId] = 0.5f * length(dropletVelocityBuf) * length(dropletVelocityBuf) / dropletMass;

    float3 newPosition = positions[dropletId] + dropletVelocity / dropletMass;
    positions[dropletId] = normalizePosition(newPosition, simulationParams.boxSizeX, 
            simulationParams.boxSizeY, simulationParams.boxSizeZ);
    velocities[dropletId] = dropletVelocity;
    velocities0[dropletId] = 2 * dropletVelocity - dropletVelocityBuf;
        
    if(dropletId == 0) {
        printf("v %e %e %e\n", 
        velocities[dropletId].x, velocities[dropletId].y, velocities[dropletId].z);
        
        printf("vbuf %e %e %e\n", 
        dropletVelocityBuf.x,dropletVelocityBuf.y,dropletVelocityBuf.z);
        
        printf("v0 %e %e %e\n", 
        velocities0[dropletId].x, velocities0[dropletId].y, velocities0[dropletId].z);
        
        printf("xxx %e %e %e\n", 
        newPosition.x,newPosition.y,newPosition.z);
    }
}

kernel void generateTube(global float3* vector, global int* types, global int* states, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    
    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
        
    float averageDropletDistance = simulationParams.averageDropletDistance;
    float boxSizeX = simulationParams.boxSizeX;
    float boxSizeY = simulationParams.boxSizeY;
    float boxSizeZ = simulationParams.boxSizeZ;
    
    int numberOfDropletsPerXDim = ceil(2 * boxSizeX / averageDropletDistance);
    int numberOfDropletsPerYDim = ceil(2 * boxSizeY / averageDropletDistance);
    int numberOfDropletsPerZDim = ceil(2 * boxSizeZ / averageDropletDistance);
    int squareOfNumberOfDropletsPerDim = numberOfDropletsPerXDim * numberOfDropletsPerYDim;
    
    int dropletIdPartX = dropletId % numberOfDropletsPerXDim;
    int dropletIdPartY = (dropletId / numberOfDropletsPerXDim) % numberOfDropletsPerYDim;
    int dropletIdPartZ = dropletId / squareOfNumberOfDropletsPerDim;
    
    float x = (dropletIdPartX + 0.5f) * averageDropletDistance - boxSizeX;
    float y = (dropletIdPartY + 0.5f) * averageDropletDistance - boxSizeY;
    float z = (dropletIdPartZ + 0.5f) * averageDropletDistance - boxSizeZ;
    
    int seed = states[dropletId];   
    float distanceFromY = sqrt(x * x + z * z);
    if (distanceFromY >= simulationParams.radiusIn) {
        types[dropletId] = 0;
    } else {
        float randomNum = rand(&seed, 1);
        types[dropletId] = (int)(randomNum / (1.0f / (simulationParams.numberOfTypes - 1))) + 1;               
    }
        
    states[dropletId] = seed;
    vector[dropletId] = (float3) (x, y, z);
}

kernel void countDropletsPerType(global int* types, global int* numberOfDropletsPerType, 
        constant SimulationParameters* simulationParameters) {

    SimulationParameters simulationParams = simulationParameters[0];
    
    int typeId = get_global_id(0);
    if (typeId >= simulationParams.numberOfTypes) {
        return;
    }
    
    int i, count = 0, numberOfDroplets = simulationParams.numberOfDroplets;
    for(i = 0; i < numberOfDroplets; ++i) {
        if(types[i] == typeId) {
            ++count;
        }
    }
    numberOfDropletsPerType[typeId] = count;
}

kernel void generateRandomPositions(global float3* vector, global int* types, global int* states, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    
    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
        
    float averageDropletDistance = simulationParams.averageDropletDistance;
    float boxSizeX = simulationParams.boxSizeX;
    float boxSizeY = simulationParams.boxSizeY;
    float boxSizeZ = simulationParams.boxSizeZ;
    
    int numberOfDropletsPerXDim = ceil(2 * boxSizeX / averageDropletDistance);
    int numberOfDropletsPerYDim = ceil(2 * boxSizeY / averageDropletDistance);
    int numberOfDropletsPerZDim = ceil(2 * boxSizeZ / averageDropletDistance);
    int squareOfNumberOfDropletsPerDim = numberOfDropletsPerXDim * numberOfDropletsPerYDim;
    
    int dropletIdPartX = dropletId % numberOfDropletsPerXDim;
    int dropletIdPartY = (dropletId / numberOfDropletsPerXDim) % numberOfDropletsPerYDim;
    int dropletIdPartZ = dropletId / squareOfNumberOfDropletsPerDim;
    
    float x = (dropletIdPartX + 0.5f) * averageDropletDistance - boxSizeX;
    float y = (dropletIdPartY + 0.5f) * averageDropletDistance - boxSizeY;
    float z = (dropletIdPartZ + 0.5f) * averageDropletDistance - boxSizeZ;
    
    int seed = states[dropletId];   
    float interval = 1.0f / simulationParams.numberOfTypes;
    float randomNum = rand(&seed, 1);
    types[dropletId] = (int)(randomNum / interval);

    states[dropletId] = seed;
    vector[dropletId] = (float3) (x, y, z);
}

kernel void generateBoryczko(global float3* vector, global int* types, global int* states,
    constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int id = get_global_id(0);
    if (id >= 1) {
        return;
    }
    
    int i, ib, ix, iy, iz, nparts_r;
    float factx, facty, factz, shftx, shfty, shftz;
    float xs, ys, zs;
    float x, y, z;
    float xmin, xmax, ymin, ymax, zmin, zmax;
    float xb[4] = {0.5, -0.5, 0.5, -0.5};
    float yb[4] = {0.5, -0.5,-0.5,  0.5};
    float zb[4] = {0.5,  0.5,-0.5, -0.5};
    
    float boxSizeX = simulationParams.boxSizeX;
    float boxSizeY = simulationParams.boxSizeY;
    float boxSizeZ = simulationParams.boxSizeZ;
    int cellsNoX = simulationParams.cellsNoX;
    int cellsNoY = simulationParams.cellsNoY;
    int cellsNoZ = simulationParams.cellsNoZ;
    
    int seed;
    float interval = 1.0f / simulationParams.numberOfTypes;
    float randomNum;
    
    xmin = 0;
    xmax = boxSizeX * 2;
    ymin = 0;
    ymax = boxSizeY * 2;
    zmin = 0;
    zmax = boxSizeZ * 2;
    
    factx = 0.5f;
    facty = 0.5f;
    factz = 0.5f;

    shftx = 2.0 * factx;
    shfty = 2.0 * facty;
    shftz = 2.0 * factz;
    
    i = 0;

    for ( ib = 0; ib < 4; ib ++ ) {
       zs = factz + zb[ib] * factz;
       for ( iz = 0; iz < cellsNoZ; iz ++ ) {
          ys = facty + yb[ib] * facty;
          for ( iy = 0; iy < cellsNoY; iy ++ ) {
             xs = factx + xb[ib] * factx;
             for ( ix = 0; ix < cellsNoX; ix ++ ) {
                if ( (xs > xmin && xs < xmax) &&
                     (ys > ymin && ys < ymax) &&
                     (zs > zmin && zs < zmax) ) {
                   vector[i].x = xs - boxSizeX;
                   vector[i].y = ys - boxSizeY;
                   vector[i].z = zs - boxSizeZ;
                
                   seed = states[i];     
                   randomNum = rand(&seed, 1);
                   types[i] = (int)(randomNum / interval);
                   states[i] = seed;
                   i++;
                }
                xs = xs + shftx;
             }
             ys = ys + shfty;
          }
          zs = zs + shftz;
       }
    }
}

kernel void generateVelocities(global float3* velocities, global float3* velocities0, global float3* forces, 
        global float* energy, global int* states, global int* types, constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];

    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    velocities[dropletId] = 0;
    velocities0[dropletId] = 0;
    forces[dropletId] = 0;
    energy[dropletId] = 0;
}

kernel void calculateAverageVelocity(global float3* velocities, global float3* velocities0, local float3* partialSums, 
        global float3* averageVelocity, constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int numberOfDroplets = simulationParams.numberOfDroplets;
        
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    float3 partialSum = 0;
    while (globalId < numberOfDroplets) {
        partialSum += velocities[globalId];
        globalId += globalSize;
    }
    
    partialSums[localId] = partialSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    int offset;
    for(offset = get_local_size(0)/2; offset > 0; offset >>= 1) {
        if(localId < offset){
            partialSums[localId] += partialSums[localId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(localId == 0) {
        averageVelocity[get_group_id(0)] = partialSums[0];
    }
}

kernel void calculateKineticEnergy(global float* dropletsEnergy, local float* partialEnergy, global float* energy, 
        global int* types, constant DropletParameters* dropletParameters,
        constant SimulationParameters* simulationParameters, int type) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int numberOfDroplets = simulationParams.numberOfDroplets;
    
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    float partialSum = 0;
    while (globalId < numberOfDroplets) {
        if(types[globalId] == type) {
            partialSum += dropletsEnergy[globalId];
        }
        globalId += globalSize;
    }
    
    partialEnergy[localId] = partialSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    int offset;
    for(offset = get_local_size(0)/2; offset > 0; offset >>= 1) {
        if(localId < offset){
            partialEnergy[localId] += partialEnergy[localId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(localId == 0) {
        energy[get_group_id(0)] = partialEnergy[0];
    }
}

kernel void fillCells(global int* cells, global float3* positions, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];

    int cellId = get_global_id(0);
    if (cellId >= simulationParams.numberOfCells) {
        return;
    }

    float boxSizeX = simulationParams.boxSizeX;
    float boxSizeY = simulationParams.boxSizeY;
    float boxSizeZ = simulationParams.boxSizeZ;
    int numberOfDroplets = simulationParams.numberOfDroplets; 
    int maxDropletsPerCell = simulationParams.maxDropletsPerCell;

    int dropletId = 0, freeId = 0;
    for(; dropletId < numberOfDroplets; ++dropletId) {
        float3 position = positions[dropletId];
        int predictedCellId = calculateCellId(position, boxSizeX, boxSizeY, boxSizeZ);
        if(predictedCellId == cellId) {
            cells[maxDropletsPerCell * cellId + freeId++] = dropletId;
        }
    }
    for(; freeId < maxDropletsPerCell; ++freeId) {
        cells[maxDropletsPerCell * cellId + freeId] = -1;
    }
}

kernel void fillCellNeighbours(global int* cellNeighbours, constant SimulationParameters* simulationParameters) {
    
    SimulationParameters simulationParams = simulationParameters[0];
    
    int cellId = get_global_id(0);
    if (cellId >= simulationParams.numberOfCells) {
        return;
    }
    
    float boxSizeX = simulationParams.boxSizeX;
    float boxSizeY = simulationParams.boxSizeY;
    float boxSizeZ = simulationParams.boxSizeZ;
    
    int numberOfCellsPerXDim = simulationParams.cellsNoX;
    int numberOfCellsPerYDim = simulationParams.cellsNoY;
    int numberOfCellsPerZDim = simulationParams.cellsNoZ;
    int squareOfNumberOfCellsPerDim = numberOfCellsPerXDim * numberOfCellsPerYDim;
    
    int cellIdPartX = cellId % numberOfCellsPerXDim;
    int cellIdPartY = (cellId / numberOfCellsPerXDim) % numberOfCellsPerYDim;
    int cellIdPartZ = cellId / squareOfNumberOfCellsPerDim;

    int cellIndex = cellId * 27;
    
    int neighboursX[3];
    int neighboursY[3];
    int neighboursZ[3];

    neighboursX[0] = cellIdPartX;
    neighboursX[1] = (cellIdPartX + numberOfCellsPerXDim - 1) % numberOfCellsPerXDim;
    neighboursX[2] = (cellIdPartX + numberOfCellsPerXDim + 1) % numberOfCellsPerXDim;
    
    neighboursY[0] = cellIdPartY;
    neighboursY[1] = (cellIdPartY + numberOfCellsPerYDim - 1) % numberOfCellsPerYDim;
    neighboursY[2] = (cellIdPartY + numberOfCellsPerYDim + 1) % numberOfCellsPerYDim;

    neighboursZ[0] = cellIdPartZ;
    neighboursZ[1] = (cellIdPartZ + numberOfCellsPerZDim - 1) % numberOfCellsPerZDim;
    neighboursZ[2] = (cellIdPartZ + numberOfCellsPerZDim + 1) % numberOfCellsPerZDim;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 3; k++){
                cellNeighbours[cellIndex++] = neighboursX[i] + neighboursY[j] * numberOfCellsPerXDim + neighboursZ[k] * squareOfNumberOfCellsPerDim;
            }
        }
    }
}
