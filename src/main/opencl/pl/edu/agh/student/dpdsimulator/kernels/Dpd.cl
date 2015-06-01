#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct SimulationParameters {
    double boxSizeX;
    double boxSizeY;
    double boxSizeZ;
    int cellsNoX;
    int cellsNoY;
    int cellsNoZ;
    double cellRadius;
    int maxDropletsPerCell;
    int numberOfDroplets;
    int numberOfCells;
    int numberOfTypes;
    double deltaTime;
    double radiusIn;
    double accelerationVesselPart;
    double accelerationValue;
    int accelerationVeselSteps;
    double averageDropletDistance;
    bool shouldSimulateVesselDroplets;
} SimulationParameters;

typedef struct DropletParameters {
    double mass;
} DropletParameters;

typedef struct PairParameters {
    double cutoffRadius;
    double pi;
    double sigma;
    double gamma;
} PairParameters;

double weightR(double distanceValue, double cutoffRadius) {
    return 1.0 / distanceValue - 1.0 / cutoffRadius;
}

double3 normalizePosition(double3 vector, double boxSizeX, double boxSizeY, double boxSizeZ) {
    double3 changeVector = (double3)(boxSizeX, boxSizeY, boxSizeZ);
    return fmod(fmod(vector + changeVector, 2.0 * changeVector) + 2.0 * changeVector, 2.0 * changeVector) - changeVector;
}

global int calculateHash(int d1, int d2) {    
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
double rand(int* seed, int step) {
/*
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = (*seed * a) % m;
    double randomValue = (double)(*seed) / m;
    if(randomValue < 0) {
        return -randomValue;
    }
    return randomValue;*/

    int    iy, ix, i;
    double zvar, fl, p;
//
    for(i = 0; i < step; i++){
        zvar = *seed;
        ix   = zvar*65539.0 / 2147483648.0;
        iy   = zvar*65539.0 - ix * 2147483648.0;
        fl   = iy;
        *seed   = iy;
    }
    return fl * 0.4656613e-09;
}

double rand2(global int* seed, global int* seed2) {      
    /*
    global int* smaller = calculateHash(seed, seed2);
    long const a = 16807L;
    long const m = 2147483647L;
    *smaller = (*smaller * a) % m;
    double randomValue = (double)(*smaller) / m;
    if(randomValue < 0) {
        return -randomValue;
    }
    return randomValue;
    */
    /*
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = (*seed * a) % m;
    double randomValue = (double)(*seed) / m;
    if(randomValue < 0) {
        return -randomValue;
    }
    return randomValue;
    */
    /*
    long const OUR_RAND_MAX = 2147483647L;
    double m;
    double P;
    int a = 1664525;
    int c = 1013904223;
    
    m = 4294967296.0;
    
    P = (*seed + *seed2) & OUR_RAND_MAX;
    P = P/m * 2.0;
    
    *seed = (a * (*seed) + c) & OUR_RAND_MAX;
    *seed2 = (a * (*seed2) + c) & OUR_RAND_MAX;
    return P;
    */
    

    int    iy, ix;
    double zvar, fl, p;
//
    zvar = *seed;
    ix   = zvar*65539.0 / 2147483648.0;
    iy   = zvar*65539.0 - ix * 2147483648.0;
    fl   = iy;
    *seed   = iy;
    return fl * 0.4656613e-09;
    
}
// <-1; 1>
double pairRandom(int dropletId, int neighbourId, int step) {
    int seed = calculateHash(dropletId, neighbourId);
    return 2 * rand(&seed, step) - 1;
    //return rand2(seed, seed2);
}

int calculateCellId(double3 position, double boxSizeX, double boxSizeY, double boxSizeZ) {
    return ((int)((position.x + boxSizeX))) + 
            ((int)(2 * boxSizeX)) * (((int)((position.y + boxSizeY))) + 
                    ((int)(2 * boxSizeY)) * ((int)((position.z + boxSizeZ))));
}

int3 calculateCellCoordinates(int dropletCellId, double boxSizeX, double boxSizeY, 
        double boxSizeZ, int cellsNoX, int cellsNoY, int cellsNoZ) {
    int dropletCellZ = dropletCellId / (cellsNoX * cellsNoY);
    int dropletCellX = dropletCellId - dropletCellZ * cellsNoX * cellsNoY;
    int dropletCellY = dropletCellX / cellsNoX;
    dropletCellX = dropletCellId % cellsNoX;
    return (int3)(dropletCellX, dropletCellY, dropletCellZ);
}

double3 getNeighbourPosition(global double3* positions, int3 dropletCellCoordinates, 
        int dropletCellId, int neighbourCellId, int neighbourId, 
        double boxSizeX, double boxSizeY, double boxSizeZ, int cellsNoX, int cellsNoY, int cellsNoZ) {
    double3 position = positions[neighbourId];
    
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

double3 calculateForce(global double3* positions, global double3* velocities, global double3* velocities0, global int* types, 
        global int* cells, global int* cellNeighbours, constant PairParameters* pairParams, constant DropletParameters* dropletParams, 
        SimulationParameters simulationParams, int dropletId, int step, global double* forces0, global int* states, global float* randoms) {
            
    int dropletType = types[dropletId];
        
    if(!simulationParams.shouldSimulateVesselDroplets
            && dropletType == 0) {
        return (double3)(0.0, 0.0, 0.0);
    }
    
    double boxSizeX = simulationParams.boxSizeX;
    double boxSizeY = simulationParams.boxSizeY;
    double boxSizeZ = simulationParams.boxSizeZ;
    int cellsNoX = simulationParams.cellsNoX;
    int cellsNoY = simulationParams.cellsNoY;
    int cellsNoZ = simulationParams.cellsNoZ;
    int maxDropletsPerCell = simulationParams.maxDropletsPerCell;
    int numberOfTypes = simulationParams.numberOfTypes;
        
    double3 conservativeForce = (double3)(0.0, 0.0, 0.0);
    double3 dissipativeForce = (double3)(0.0, 0.0, 0.0);
    double3 randomForce = (double3)(0.0, 0.0, 0.0);
    
    double3 dropletPosition = positions[dropletId];
    double3 dropletVelocity = velocities0[dropletId];    
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
                
                double3 neighbourPosition = getNeighbourPosition(positions, dropletCellCoordinates,
                        dropletCellId, cellId, neighbourId, boxSizeX, boxSizeY, boxSizeZ, cellsNoX, cellsNoY, cellsNoZ);
                double distanceValue = distance(neighbourPosition, dropletPosition);

                PairParameters pairParameters = pairParams[dropletType * numberOfTypes + neighbourType];
                double cutoffRadius = pairParameters.cutoffRadius;

                if(distanceValue < cutoffRadius) {
                    double3 normalizedPositionVector = neighbourPosition - dropletPosition;
                    double weightRValue = weightR(distanceValue, cutoffRadius);
                    double weightDValue = weightRValue * weightRValue;

                    double pi = pairParameters.pi;
                    double gamma = pairParameters.gamma;
                    double sigma = pairParameters.sigma;

                    conservativeForce += pi * weightRValue * normalizedPositionVector;

                    dissipativeForce -= gamma * weightDValue * normalizedPositionVector
                            * dot(velocities0[neighbourId] - dropletVelocity, normalizedPositionVector);

                    int min, max;
                    if(dropletId < neighbourId){
                        min = dropletId;
                        max = neighbourId;
                    } else {
                        min = neighbourId;
                        max = dropletId;
                    }

                    randomForce += sigma * weightRValue * normalizedPositionVector 
                        * randoms[simulationParams.numberOfDroplets * min + max - (min * (min + 1))/2];                        
                    ++noOfNeighbours;
                }
            }
        }
    }
    
    double3 force;
    if(dropletType != 0 
            && dropletPosition.y < simulationParams.accelerationVesselPart 
            && step < simulationParams.accelerationVeselSteps) {
        force = conservativeForce + dissipativeForce + randomForce + (double3)(0, -simulationParams.accelerationValue, 0);
    } else {
        force = conservativeForce + dissipativeForce + randomForce;
    }
    
    return force / (noOfNeighbours + 1);
}

kernel void calculateForces(global double3* positions, global double3* velocities, global double3* velocities0, global double3* forces, 
        global double* forces0, global int* types, global int* cells, global int* cellNeighbours, constant PairParameters* pairParams, 
        constant DropletParameters* dropletParams, constant SimulationParameters* simulationParameters, int step, global int* states, global float* randoms) {

    SimulationParameters simulationParams = simulationParameters[0];

    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    forces[dropletId] = calculateForce(positions, velocities, velocities0, types, cells, 
            cellNeighbours, pairParams, dropletParams, simulationParams, dropletId, step, forces0, states, randoms);
}

kernel void calculateNewPositionsAndVelocities(global double3* positions, global double3* velocities,
        global double3* velocities0, global double3* forces, global double* dropletsEnergy, global int* types,
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
        
    double3 dropletForce = forces[dropletId];
    double3 dropletVelocity = velocities[dropletId] - dropletForce;
    double3 dropletVelocityBuf = 0.5 * (velocities[dropletId] + dropletVelocity);
    double dropletMass = dropletParams[types[dropletId]].mass;
    
    dropletsEnergy[dropletId] = 0.5 * length(dropletVelocityBuf) * length(dropletVelocityBuf) / dropletMass;

    double3 newPosition = positions[dropletId] + dropletVelocity / dropletMass;
    positions[dropletId] = normalizePosition(newPosition, simulationParams.boxSizeX, 
            simulationParams.boxSizeY, simulationParams.boxSizeZ);
    velocities[dropletId] = dropletVelocity;
    velocities0[dropletId] = 2 * dropletVelocity - dropletVelocityBuf;
        
    /*if(dropletId == 0) {
        printf("v %e %e %e\n", 
        velocities[dropletId].x, velocities[dropletId].y, velocities[dropletId].z);
        
        printf("vbuf %e %e %e\n", 
        dropletVelocityBuf.x,dropletVelocityBuf.y,dropletVelocityBuf.z);
        
        printf("v0 %e %e %e\n", 
        velocities0[dropletId].x, velocities0[dropletId].y, velocities0[dropletId].z);
        
        printf("xxx %e %e %e\n", 
        newPosition.x,newPosition.y,newPosition.z);
    }*/
}

kernel void generateTube(global double3* vector, global int* types, global int* states, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int id = get_global_id(0);
    if (id >= 1) {
        return;
    }
    
    int i, ib, ix, iy, iz, nparts_r;
    double factx, facty, factz, shftx, shfty, shftz;
    double xs, ys, zs;
    double x, y, z;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double xb[4] = {0.5, -0.5, 0.5, -0.5};
    double yb[4] = {0.5, -0.5,-0.5,  0.5};
    double zb[4] = {0.5,  0.5,-0.5, -0.5};
    
    double boxSizeX = simulationParams.boxSizeX;
    double boxSizeY = simulationParams.boxSizeY;
    double boxSizeZ = simulationParams.boxSizeZ;
    int cellsNoX = simulationParams.cellsNoX;
    int cellsNoY = simulationParams.cellsNoY;
    int cellsNoZ = simulationParams.cellsNoZ;
    
    int seed;
    double interval = 1.0 / simulationParams.numberOfTypes;
    double randomNum;
    
    xmin = 0;
    xmax = boxSizeX * 2;
    ymin = 0;
    ymax = boxSizeY * 2;
    zmin = 0;
    zmax = boxSizeZ * 2;
    
    factx = 0.5;
    facty = 0.5;
    factz = 0.5;

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
                   double distanceFromY = sqrt(vector[i].x * vector[i].x + vector[i].z * vector[i].z);
                   if (distanceFromY >= simulationParams.radiusIn) {
                       types[i] = 0;
                   } else {
                       double randomNum = rand(&seed, 1);
                       types[i] = (int)(randomNum / (1.0 / (simulationParams.numberOfTypes - 1))) + 1;               
                   }
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

kernel void generateRandomPositions(global double3* vector, global int* types, global int* states, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    
    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
        
    double averageDropletDistance = simulationParams.averageDropletDistance;
    double boxSizeX = simulationParams.boxSizeX;
    double boxSizeY = simulationParams.boxSizeY;
    double boxSizeZ = simulationParams.boxSizeZ;
    
    int numberOfDropletsPerXDim = ceil(2 * boxSizeX / averageDropletDistance);
    int numberOfDropletsPerYDim = ceil(2 * boxSizeY / averageDropletDistance);
    int numberOfDropletsPerZDim = ceil(2 * boxSizeZ / averageDropletDistance);
    int squareOfNumberOfDropletsPerDim = numberOfDropletsPerXDim * numberOfDropletsPerYDim;
    
    int dropletIdPartX = dropletId % numberOfDropletsPerXDim;
    int dropletIdPartY = (dropletId / numberOfDropletsPerXDim) % numberOfDropletsPerYDim;
    int dropletIdPartZ = dropletId / squareOfNumberOfDropletsPerDim;
    
    double x = (dropletIdPartX + 0.5) * averageDropletDistance - boxSizeX;
    double y = (dropletIdPartY + 0.5) * averageDropletDistance - boxSizeY;
    double z = (dropletIdPartZ + 0.5) * averageDropletDistance - boxSizeZ;
    
    int seed = states[dropletId];   
    double interval = 1.0 / simulationParams.numberOfTypes;
    double randomNum = rand(&seed, 1);
    types[dropletId] = (int)(randomNum / interval);

    states[dropletId] = seed;
    vector[dropletId] = (double3) (x, y, z);
}

kernel void generateBoryczko(global double3* vector, global int* types, global int* states,
    constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int id = get_global_id(0);
    if (id >= 1) {
        return;
    }
    
    int i, ib, ix, iy, iz, nparts_r;
    double factx, facty, factz, shftx, shfty, shftz;
    double xs, ys, zs;
    double x, y, z;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double xb[4] = {0.5, -0.5, 0.5, -0.5};
    double yb[4] = {0.5, -0.5,-0.5,  0.5};
    double zb[4] = {0.5,  0.5,-0.5, -0.5};
    
    double boxSizeX = simulationParams.boxSizeX;
    double boxSizeY = simulationParams.boxSizeY;
    double boxSizeZ = simulationParams.boxSizeZ;
    int cellsNoX = simulationParams.cellsNoX;
    int cellsNoY = simulationParams.cellsNoY;
    int cellsNoZ = simulationParams.cellsNoZ;
    
    int seed;
    double interval = 1.0 / simulationParams.numberOfTypes;
    double randomNum;
    
    xmin = 0;
    xmax = boxSizeX * 2;
    ymin = 0;
    ymax = boxSizeY * 2;
    zmin = 0;
    zmax = boxSizeZ * 2;
    
    factx = 0.5;
    facty = 0.5;
    factz = 0.5;

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

kernel void generateVelocities(global double3* velocities, global double3* velocities0, global double3* forces, 
        global double* energy, global int* states, global int* types, constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];

    int dropletId = get_global_id(0);
    if (dropletId >= simulationParams.numberOfDroplets) {
        return;
    }
    
    velocities[dropletId].x = 0;
    velocities[dropletId].y = 0.000002;
    velocities[dropletId].z = 0;
    velocities0[dropletId].x = 0;
    velocities0[dropletId].y = 0.000002;
    velocities0[dropletId].z = 0;
    forces[dropletId] = 0;
    energy[dropletId] = 0;
}

kernel void calculateAverageVelocity(global double3* velocities, global double3* velocities0, local double3* partialSums, 
        global double3* averageVelocity, constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int numberOfDroplets = simulationParams.numberOfDroplets;
        
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    double3 partialSum = 0;
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

kernel void calculateKineticEnergy(global double* dropletsEnergy, local double* partialEnergy, global double* energy, 
        global int* types, constant DropletParameters* dropletParameters,
        constant SimulationParameters* simulationParameters, int type) {
            
    SimulationParameters simulationParams = simulationParameters[0];
    int numberOfDroplets = simulationParams.numberOfDroplets;
    
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    
    double partialSum = 0;
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

kernel void fillCells(global int* cells, global double3* positions, 
        constant SimulationParameters* simulationParameters) {
            
    SimulationParameters simulationParams = simulationParameters[0];

    int cellId = get_global_id(0);
    if (cellId >= simulationParams.numberOfCells) {
        return;
    }

    double boxSizeX = simulationParams.boxSizeX;
    double boxSizeY = simulationParams.boxSizeY;
    double boxSizeZ = simulationParams.boxSizeZ;
    int numberOfDroplets = simulationParams.numberOfDroplets; 
    int maxDropletsPerCell = simulationParams.maxDropletsPerCell;

    int dropletId = 0, freeId = 0;
    for(; dropletId < numberOfDroplets; ++dropletId) {
        double3 position = positions[dropletId];
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
    
    double boxSizeX = simulationParams.boxSizeX;
    double boxSizeY = simulationParams.boxSizeY;
    double boxSizeZ = simulationParams.boxSizeZ;
    
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
