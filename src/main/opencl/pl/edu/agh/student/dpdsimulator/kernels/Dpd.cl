typedef struct SimulationParameters {
    float boxSize;
    float boxWidth;
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
            position.x -= 2 * boxSize;
        }
        
        if(dropletCellCoordinates.x == cellsNoXZ - 1 && neighbourCellCoordinates.x == 0) {
            position.x += 2 * boxSize;
        }
        
        if(dropletCellCoordinates.y == 0 && neighbourCellCoordinates.y == cellsNoY - 1) {
            position.y -= 2 * boxWidth;
        }
        
        if(dropletCellCoordinates.y == cellsNoY - 1 && neighbourCellCoordinates.y == 0) {
            position.y += 2 * boxWidth;
        }
        
        if(dropletCellCoordinates.z == 0 && neighbourCellCoordinates.z == cellsNoXZ - 1) {
            position.z -= 2 * boxSize;
        }
        
        if(dropletCellCoordinates.z == cellsNoXZ - 1 && neighbourCellCoordinates.z == 0) {
            position.z += 2 * boxSize;
        }
    }
    
    return position;
}

float3 calculateForce(global float3* positions, global float3* velocities, global int* types, global int* cells, 
        global int* cellNeighbours, constant PairParameters* pairParams, constant DropletParameters* dropletParams, 
        SimulationParameters simulationParams, int dropletId, int step) {
            
    int dropletType = types[dropletId];
        
    if(!simulationParams.shouldSimulateVesselDroplets
            && dropletType == 0) {
        return (float3)(0.0, 0.0, 0.0);
    }
    
    float cellRadius = simulationParams.cellRadius;
    float boxSize = simulationParams.boxSize;
    float boxWidth = simulationParams.boxWidth;
    int maxDropletsPerCell = simulationParams.maxDropletsPerCell;
    int numberOfTypes = simulationParams.numberOfTypes;
        
    float3 conservativeForce = (float3)(0.0, 0.0, 0.0);
    float3 dissipativeForce = (float3)(0.0, 0.0, 0.0);
    float3 randomForce = (float3)(0.0, 0.0, 0.0);
    
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
                int neighbourType = types[neighbourId];
                
                float3 neighbourPosition = getNeighbourPosition(positions, dropletCellCoordinates, 
                        cellRadius, dropletCellId, cellId, neighbourId, boxSize, boxWidth, cellsNoXZ, cellsNoY);
                float distanceValue = distance(neighbourPosition, dropletPosition);

                PairParameters pairParameters = pairParams[dropletType * numberOfTypes + neighbourType];
                float cutoffRadius = pairParameters.cutoffRadius;

                if(distanceValue < cutoffRadius) {
                    float3 normalizedPositionVector = normalize(neighbourPosition - dropletPosition);
                    float weightRValue = weightR(distanceValue, cutoffRadius);
                    float weightDValue = weightRValue * weightRValue;

                    float pi = pairParameters.pi;
                    float gamma = pairParameters.gamma;
                    float sigma = pairParameters.sigma;

                    conservativeForce += pi * (1.0f - distanceValue / cutoffRadius) * normalizedPositionVector;

                    dissipativeForce -= gamma * weightDValue * normalizedPositionVector
                            * dot(velocities[neighbourId] - dropletVelocity, normalizedPositionVector);

                    randomForce += sigma * weightRValue * normalizedPositionVector
                            * gaussianRandom(dropletId, neighbourId, step);
                }
            }
        }
    }
    
    if(dropletType != 0 
            && dropletPosition.y < simulationParams.accelerationVesselPart && step < simulationParams.accelerationVeselSteps) {
        return conservativeForce + dissipativeForce + randomForce + (float3)(0, simulationParams.accelerationValue, 0);
    }
    return conservativeForce + dissipativeForce + randomForce;
}

kernel void calculateForces(global float3* positions, global float3* velocities, global float3* forces, 
        global int* types, global int* cells, global int* cellNeighbours, constant PairParameters* pairParams, 
        constant DropletParameters* dropletParams, constant SimulationParameters* simulationParameters, int step) {

    SimulationParameters simulationParams = simulationParameters[0];

    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    forces[dropletId] = calculateForce(positions, velocities, types, cells, 
            cellNeighbours, pairParams, dropletParams, simulationParams, dropletId, step);
}

kernel void calculateNewPositionsAndVelocities(global float3* positions, global float3* velocities,
        global float3* forces, global int* types, constant PairParameters* pairParams, 
        constant DropletParameters* dropletParams, constant SimulationParameters* simulationParameters) {

    SimulationParameters simulationParams = simulationParameters[0];
    
    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    if(!simulationParams.shouldSimulateVesselDroplets 
            && types[dropletId] == 0) {
        return;
    }

    float deltaTime = simulationParams.deltaTime;
    
    float3 dropletVelocity = velocities[dropletId];
    float3 dropletForce = forces[dropletId];
    float dropletMass = dropletParams[types[dropletId]].mass;

    float3 newPosition = positions[dropletId] + deltaTime * dropletVelocity;
    positions[dropletId] = normalizePosition(newPosition, simulationParams.boxSize, simulationParams.boxWidth);    
    velocities[dropletId] = velocities[dropletId] + deltaTime * forces[dropletId] / dropletMass;
}

kernel void generateTube(global float3* vector, global int* types, global int* states, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    
    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
        
    float averageDropletDistance = simulationParams.averageDropletDistance;
    float boxSize = simulationParams.boxSize;
    float boxWidth = simulationParams.boxWidth;
    
    int numberOfDropletsPerXZDim = ceil(2 * boxSize / averageDropletDistance);
    int numberOfDropletsPerYDim = ceil(2 * boxWidth / averageDropletDistance);
    int squareOfNumberOfDropletsPerDim = numberOfDropletsPerXZDim * numberOfDropletsPerYDim;
    
    int dropletIdPartX = dropletId % numberOfDropletsPerXZDim;
    int dropletIdPartY = (dropletId / numberOfDropletsPerXZDim) % numberOfDropletsPerYDim;
    int dropletIdPartZ = dropletId / squareOfNumberOfDropletsPerDim;
    
    float x = (dropletIdPartX + 0.5f) * averageDropletDistance - boxSize;
    float y = (dropletIdPartY + 0.5f) * averageDropletDistance - boxWidth;
    float z = (dropletIdPartZ + 0.5f) * averageDropletDistance - boxSize;
    
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

kernel void generateRandomPositions(global float3* vector, global int* types, global int* states, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    
    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
        
    float averageDropletDistance = simulationParams.averageDropletDistance;
    float boxSize = simulationParams.boxSize;
    float boxWidth = simulationParams.boxWidth;
    
    int numberOfDropletsPerXZDim = ceil(2 * boxSize / averageDropletDistance);
    int numberOfDropletsPerYDim = ceil(2 * boxWidth / averageDropletDistance);
    int squareOfNumberOfDropletsPerDim = numberOfDropletsPerXZDim * numberOfDropletsPerYDim;
    
    int dropletIdPartX = dropletId % numberOfDropletsPerXZDim;
    int dropletIdPartY = (dropletId / numberOfDropletsPerXZDim) % numberOfDropletsPerYDim;
    int dropletIdPartZ = dropletId / squareOfNumberOfDropletsPerDim;
    
    float x = (dropletIdPartX + 0.5f) * averageDropletDistance - boxSize;
    float y = (dropletIdPartY + 0.5f) * averageDropletDistance - boxWidth;
    float z = (dropletIdPartZ + 0.5f) * averageDropletDistance - boxSize;
    
    int seed = states[dropletId];   
    float interval = 1.0f / simulationParams.numberOfTypes;
    float randomNum = rand(&seed, 1);
    types[dropletId] = (int)(randomNum / interval);    
        
    states[dropletId] = seed;
    vector[dropletId] = (float3) (x, y, z);
}

kernel void generateVelocities(global float3* velocities, global int* states, global int* types, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];

    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    velocities[dropletId] = (float3) (0, 0, 0);
}

kernel void calculateAverageVelocity(global float3* velocities, local float3* partialSums, 
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
    for(offset = get_local_size(0)/2; offset > 0; offset >>= 1){
        if(localId < offset){
            partialSums[localId] += partialSums[localId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(localId == 0){
        averageVelocity[get_group_id(0)] = partialSums[0];
    }
}

kernel void calculateVelocitiesEnergy(global float3* velocities, local float* partialEnergy, 
        global float* energy, constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int numberOfDroplets = simulationParams.numberOfDroplets;
    
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    float partialSum = 0;
    while (globalId < numberOfDroplets) {
        float3 velocity = velocities[globalId];
        partialSum += velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
        globalId += globalSize;
    }
    
    partialEnergy[localId] = partialSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    int offset;
    for(offset = get_local_size(0)/2; offset > 0; offset >>= 1){
        if(localId < offset){
            partialEnergy[localId] += partialEnergy[localId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(localId == 0){
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

    float cellRadius = simulationParams.cellRadius;
    float boxSize = simulationParams.boxSize;
    float boxWidth = simulationParams.boxWidth; 
    int numberOfDroplets = simulationParams.numberOfDroplets; 
    int maxDropletsPerCell = simulationParams.maxDropletsPerCell;

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

kernel void fillCellNeighbours(global int* cellNeighbours, constant SimulationParameters* simulationParameters) {
    
    SimulationParameters simulationParams = simulationParameters[0];
    
    int cellId = get_global_id(0);
    if (cellId >= simulationParams.numberOfCells) {
        return;
    }
    
    float cellRadius = simulationParams.cellRadius;
    float boxSize = simulationParams.boxSize;
    float boxWidth = simulationParams.boxWidth;
    
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
