package pl.edu.agh.student.dpdsimulator;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import static pl.edu.agh.student.dpdsimulator.Simulation.NANOS_IN_SECOND;
import static pl.edu.agh.student.dpdsimulator.Simulation.numberOfSteps;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public class JavaDpdMock extends Simulation {
    
    private Random random;
    private int[] cells;
    private int[] cellNeighbours;
    private float[][] positions;
    private float[][] newPositions;
    private float[][] velocities;
    private float[][] predictedVelocities;
    private float[][] newVelocities;
    private float[][] forces;
    private int[] types;
    private int[] states;
    private Dpd.DropletParameter[] dropletParameters;
    private int step;

    @Override
    public void initData(float boxSize, float boxWidth, int numberOfDropletsParam) throws IOException {
        random = new Random();
        cells = new int[maxDropletsPerCell * numberOfCells];
        cellNeighbours = new int[numberOfCells * numberOfCellNeighbours];
        positions = new float[numberOfDroplets][VEC_SIZE];
        newPositions = new float[numberOfDroplets][VEC_SIZE];
        velocities = new float[numberOfDroplets][VEC_SIZE];
        predictedVelocities = new float[numberOfDroplets][VEC_SIZE];
        newVelocities = new float[numberOfDroplets][VEC_SIZE];
        forces = new float[numberOfDroplets][VEC_SIZE];
        types = new int[numberOfDroplets];
        states = new int[numberOfDroplets];
    }    
    
    @Override
    public void performSimulation() {        
        long startTime = System.nanoTime();
        step = 0;
        initSimulationData();
        long endInitTime = System.nanoTime();
        for (step = 1; step <= numberOfSteps; ++step) {
            System.out.println("\nStep: " + step);
            performSingleStep();
            printAverageVelocity();
            swapPositions();
            swapVelocities();
        }
        long endTime = System.nanoTime();
        System.out.println("Init time: " + (endInitTime - startTime) / NANOS_IN_SECOND);
        System.out.println("Mean step time: " + (endTime - startTime) / NANOS_IN_SECOND / numberOfSteps);
    }
    
    private void initSimulationData() {
        initDropletParameters();
        initStates();
        initPositionsAndVelocities();
        fillCells(positions);
        fillCellNeighbours();
    }

    private void initDropletParameters() {
        List<Dpd.DropletParameter> parameters = super.createDropletParameters();
        dropletParameters = parameters.toArray(new Dpd.DropletParameter[parameters.size()]);
    }

    private void initStates(){
        for(int i = 0; i < numberOfDroplets; i++){
            states[i] = random.nextInt(Integer.MAX_VALUE);
        }
    }
    
    private void initPositionsAndVelocities() {
        generateTube(positions, types, states, numberOfDroplets, radiusIn, boxSize, initBoxWidth);
        generateRandomVector(velocities, states, types, thermalVelocity, flowVelocity, numberOfDroplets);
    }
    
    void generateTube(float[][] vector, int[] types, int[] states, int numberOfDroplets, 
            float radiusIn, float boxSize, float boxWidth) {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            int seed = states[dropletId];   

            float x = (rand(seed, 1) * 2 - 1) * boxSize;
            float y = (rand(seed, 1) * 2 - 1) * boxWidth;
            float z = (rand(seed, 1) * 2 - 1) * boxSize;

            float distanceFromY = (float) Math.sqrt(x * x + z * z);
            if (distanceFromY >= radiusIn) {
                types[dropletId] = 0;
            } else {
                float randomNum = rand(seed, 1);
                if (randomNum >= 0.5f) {
                    types[dropletId] = 1;    
                } else {
                    types[dropletId] = 2;
                }        
            }

            states[dropletId] = seed;
            vector[dropletId][0] = x;
            vector[dropletId][1] = y;
            vector[dropletId][2] = z;
        }
    }
    
    void generateRandomVector(float[][] vector, int[] states, int[] types, float thermalVelocity, float flowVelocity, int numberOfDroplets) {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            float x, y, z;
            int seed = states[dropletId];
            if(types[dropletId] == 0){
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
            } else {
                x = (rand(seed, 1) * 2 - 1) * thermalVelocity;
                y = ((rand(seed, 1) * 2 - 1) * flowVelocity + flowVelocity) / 2.0f;
                z = (rand(seed, 1) * 2 - 1) * thermalVelocity;
            }
            states[dropletId] = seed;
            vector[dropletId][0] = x;
            vector[dropletId][1] = y;
            vector[dropletId][2] = z;
        }
    }
    
    private void fillCells(float[][] positions) {
        fillCells(cells, positions);
    }
    
    private void fillCellNeighbours() {
        fillCellNeighbours(cellNeighbours);
    }

    private void performSingleStep() {
        calculateForces();
        calculateNewPositionsAndPredictedVelocities();
        fillCells(newPositions);
        calculateNewVelocities();
    }
    
    private void calculateForces() {
        for(int dropletId = 0; dropletId < numberOfDroplets * numberOfCellNeighbours; ++dropletId) {
            calculateForces(positions, velocities, forces, dropletParameters, types, cells, cellNeighbours, step, dropletId);
        }
    }

    private void calculateNewPositionsAndPredictedVelocities() {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            calculateNewPositionsAndPredictedVelocities(positions, velocities, forces, newPositions,
                    predictedVelocities, dropletId);
        }
    }

    private void calculateNewVelocities() {
        for(int dropletId = 0; dropletId < numberOfDroplets * numberOfCellNeighbours; ++dropletId) {
            calculateNewVelocities(newPositions, velocities, predictedVelocities, newVelocities, forces, dropletParameters, 
                    types, cells, cellNeighbours, step, dropletId);
        }
    }

    private void swapPositions() {
        float[][] tmp = positions;
        positions = newPositions;
        newPositions = tmp;
    }

    private void swapVelocities() {
        float[][] tmp = velocities;
        velocities = newVelocities;
        newVelocities = tmp;
    }

    public static final int VEC_SIZE = 3;

    public static float weightR(float dist, float cutoffRadius) {
        if (dist > cutoffRadius) {
            return 0.0f;
        }
        return 1.0f - dist / cutoffRadius;
    }

    public static float weightD(float dist, float cutoffRadius) {
        float weight = weightR(dist, cutoffRadius);
        return weight * weight;
    }

    private static float[] normalize(float[] position, float[] position1) {
        float[] result = new float[VEC_SIZE];
        float dist = distance(position, position1);
        for (int k = 0; k < VEC_SIZE; ++k) {
            result[k] = (position[k] - position1[k]) / dist;
        }
        return result;
    }

    private static float distance(float[] position, float[] position1) {
        float sum = 0.0f;
        for (int k = 0; k < VEC_SIZE; ++k) {
            sum += (float) Math.pow(position[k] - position1[k], 2);
        }
        return (float) Math.sqrt(sum);
    }

    private static float dot(float[] vector1, float[] vector2) {
        float sum = 0.0f;
        for (int k = 0; k < VEC_SIZE; ++k) {
            sum += vector1[k] * vector2[k];
        }
        return sum;
    }

    private static float[] diff(float[] vector1, float[] vector2) {
        float[] result = new float[VEC_SIZE];
        for (int k = 0; k < VEC_SIZE; ++k) {
            result[k] = vector1[k] - vector2[k];
        }
        return result;
    }

    private static float[] fmod(float[] vector, float[] factor) {
        float[] result = new float[VEC_SIZE];
        for (int k = 0; k < VEC_SIZE; ++k) {
            result[k] = vector[k] - factor[k] * ((int) (vector[k] / factor[k]));
        }

        return result;
    }

    static float[] normalizePosition(float[] vector) {
        float[] factor = new float[]{boxSize, initBoxWidth, boxSize};
        float[] factor1 = new float[]{2f * boxSize, 2f * initBoxWidth, 2f * boxSize};
        float[] factor2 = new float[]{-boxSize, -initBoxWidth, -boxSize};
        return add(fmod(add(fmod(add(vector, factor), factor1), factor1), factor1), factor2);
    }

    private static float[] add(float[] vector, float[] vector2) {
        float[] vec = new float[VEC_SIZE];
        for (int i = 0; i < VEC_SIZE; ++i) {
            vec[i] = vector[i] + vector2[i];
        }
        return vec;
    }
    
    public static float[] calculateForce(float[][] positions, float[][] velocities, Dpd.DropletParameter[] params,
        int[] types, int[] cells, int[] cellNeighbours, int dropletId, int dropletCellNeighbourId, int step) {

        float[] conservativeForce = new float[]{0.0f, 0.0f, 0.0f};
        float[] dissipativeForce = new float[]{0.0f, 0.0f, 0.0f};
        float[] randomForce = new float[]{0.0f, 0.0f, 0.0f};

        float[] dropletPosition = positions[dropletId];
        float[] dropletVelocity = velocities[dropletId];

        int dropletType = types[dropletId];
        Dpd.DropletParameter dropletParameter = params[dropletType];

        float repulsionParameter = dropletParameter.repulsionParameter();
        float gamma = dropletParameter.gamma();
        float sigma = dropletParameter.sigma();

        int dropletCellId = calculateCellId(dropletPosition);

        int cellId = cellNeighbours[dropletCellId * numberOfCellNeighbours + dropletCellNeighbourId];
        if(cellId >= 0) {
        int neighbourId = 0, j;
        for(j = 0, neighbourId = cells[maxDropletsPerCell * cellId]; neighbourId >= 0; neighbourId = cells[maxDropletsPerCell * cellId + ++j]) {
            if(neighbourId != dropletId) {
                float[] neighbourPosition = positions[neighbourId];
                float distanceValue = distance(neighbourPosition, dropletPosition);

                int neighbourType = types[neighbourId];
                Dpd.DropletParameter neighbourParameter = params[neighbourType];
                float cutoffRadius = neighbourParameter.cutoffRadius();

                if(distanceValue < cutoffRadius) {
                    float[] normalizedPositionVector = normalize(neighbourPosition, dropletPosition);
                    if(dropletType != 0 || neighbourType != 0) {
                        float weightRValue = weightR(distanceValue, cutoffRadius);
                        float weightDValue = weightRValue * weightRValue;

                        for(int k = 0; k < VEC_SIZE; ++k) {
                            conservativeForce[k] -= Math.sqrt(repulsionParameter * neighbourParameter.repulsionParameter())
                                    * (1.0f - distanceValue / cutoffRadius) * normalizedPositionVector[k];

                            dissipativeForce[k] += gamma * weightDValue * normalizedPositionVector[k]
                                    * dot(normalizedPositionVector, diff(velocities[neighbourId], dropletVelocity));

                            randomForce[k] -= sigma * weightRValue * normalizedPositionVector[k]
                                    * gaussianRandom(dropletId, neighbourId, step);
                        }
                    }
                }
            }
        }
        }

        float[] result = new float[VEC_SIZE];
        for (int k = 0; k < VEC_SIZE; ++k) {
            result[k] = conservativeForce[k] + dissipativeForce[k] + randomForce[k];
        }
        return result;
    }
        
    public static void fillCells(int[] cells, float[][] positions) {
        for(int cellId = 0; cellId < numberOfCells; ++cellId) {
            int dropletId = 0, freeId = 0;
            for(; dropletId < numberOfDroplets; ++dropletId) {
                float[] position = positions[dropletId];
                int predictedCellId = calculateCellId(position);
                if(predictedCellId == cellId) {
                    cells[maxDropletsPerCell * cellId + freeId++] = dropletId;
                }
            }
            for(; freeId < maxDropletsPerCell; ++freeId) {
                cells[maxDropletsPerCell * cellId + freeId] = -1;
            }
        }
    }

    public static void fillCellNeighbours(int[] cellNeighbours) {
        for(int cellId = 0; cellId < numberOfCells; ++cellId) {
            int numberOfCellsPerXZDim = (int) Math.ceil(2 * boxSize / cellRadius);
            int numberOfCellsPerYDim = (int) Math.ceil(2 * initBoxWidth / cellRadius);
            int squareOfNumberOfCellsPerDim = numberOfCellsPerXZDim * numberOfCellsPerYDim;

            int cellIdPartX = cellId % numberOfCellsPerXZDim;
            int cellIdPartY = (cellId / numberOfCellsPerXZDim) % numberOfCellsPerYDim;
            int cellIdPartZ = cellId / squareOfNumberOfCellsPerDim;

            int cellIndex = cellId * numberOfCellNeighbours;
            cellNeighbours[cellIndex++] = cellId;

            if(cellIdPartX > 0) {
                cellNeighbours[cellIndex++] = cellId - 1;
            }

            if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                cellNeighbours[cellIndex++] = cellId + 1;
            }

            if(cellIdPartY > 0) {
                cellNeighbours[cellIndex++] = cellId - numberOfCellsPerXZDim;

                if(cellIdPartX > 0) {
                    cellNeighbours[cellIndex++] = cellId - numberOfCellsPerXZDim - 1;
                }

                if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                    cellNeighbours[cellIndex++] = cellId - numberOfCellsPerXZDim + 1;
                }
            }

            if(cellIdPartY < numberOfCellsPerYDim - 1) {
                cellNeighbours[cellIndex++] = cellId + numberOfCellsPerXZDim;

                if(cellIdPartX > 0) {
                    cellNeighbours[cellIndex++] = cellId + numberOfCellsPerXZDim - 1;
                }

                if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                    cellNeighbours[cellIndex++] = cellId + numberOfCellsPerXZDim + 1;
                }
            }

            if(cellIdPartZ > 0) {
                cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim;

                if(cellIdPartX > 0) {
                    cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - 1;
                }

                if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                    cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + 1;
                }

                if(cellIdPartY > 0) {
                    cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - numberOfCellsPerXZDim;

                    if(cellIdPartX > 0) {
                        cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - numberOfCellsPerXZDim - 1;
                    }

                    if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                        cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim - numberOfCellsPerXZDim + 1;
                    }
                }

                if(cellIdPartY < numberOfCellsPerYDim - 1) {
                    cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + numberOfCellsPerXZDim;

                    if(cellIdPartX > 0) {
                        cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + numberOfCellsPerXZDim - 1;
                    }

                    if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                        cellNeighbours[cellIndex++] = cellId - squareOfNumberOfCellsPerDim + numberOfCellsPerXZDim + 1;
                    }
                }
            }

            if(cellIdPartZ < numberOfCellsPerXZDim - 1) {
                cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim;

                if(cellIdPartX > 0) {
                    cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - 1;
                }

                if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                    cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + 1;
                }

                if(cellIdPartY > 0) {
                    cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - numberOfCellsPerXZDim;

                    if(cellIdPartX > 0) {
                        cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - numberOfCellsPerXZDim - 1;
                    }

                    if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                        cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim - numberOfCellsPerXZDim + 1;
                    }
                }

                if(cellIdPartY < numberOfCellsPerYDim - 1) {
                    cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + numberOfCellsPerXZDim;

                    if(cellIdPartX > 0) {
                        cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + numberOfCellsPerXZDim - 1;
                    }

                    if(cellIdPartX < numberOfCellsPerXZDim - 1) {
                        cellNeighbours[cellIndex++] = cellId + squareOfNumberOfCellsPerDim + numberOfCellsPerXZDim + 1;
                    }
                }
            }

            for(int n = (cellId + 1) * numberOfCellNeighbours; cellIndex < n; ++cellIndex) {
                cellNeighbours[cellIndex] = -1;
            }
        }
    }
        
    public static void calculateForces(float[][] positions, float[][] velocities, float[][] forces,
            Dpd.DropletParameter[] params, int[] types, int[] cells, int[] cellNeighbours, int step, int fullDropletId) {
        
        int dropletId = fullDropletId / numberOfCellNeighbours;
        if (dropletId >= numberOfDroplets) {
            return;
        }
    
        float localForces[][] = new float[numberOfCellNeighbours][VEC_SIZE];
        for(int dropletCellNeighbourId = 0; dropletCellNeighbourId < numberOfCellNeighbours; ++dropletCellNeighbourId) {
            float[] tmp = calculateForce(positions, velocities, params, 
                    types, cells, cellNeighbours, dropletId, dropletCellNeighbourId, step);
            for(int j = 0; j < VEC_SIZE; ++j) {
                localForces[dropletCellNeighbourId][j] = tmp[j];
            }
        }
        
        for(int j = 0; j < VEC_SIZE; ++j) {
            forces[dropletId][j] = 0;
        }
        for(int i = 0; i < numberOfCellNeighbours; ++i) {
            for(int j = 0; j < VEC_SIZE; ++j) {
                forces[dropletId][j] += localForces[i][j];
            }
        }
    }

    public static void calculateNewPositionsAndPredictedVelocities(float[][] positions, float[][] velocities,
            float[][] forces, float[][] newPositions, float[][] predictedVelocities, int dropletId) {

        float[] newPosition = new float[VEC_SIZE];
        for (int i = 0; i < VEC_SIZE; ++i) {
            newPosition[i] = positions[dropletId][i]
                    + deltaTime * velocities[dropletId][i] + 0.5f * deltaTime * deltaTime * forces[dropletId][i];
            predictedVelocities[dropletId][i] = velocities[dropletId][i] + lambda * deltaTime * forces[dropletId][i];
        }
        float[] normalized = normalizePosition(newPosition);
        for (int i = 0; i < VEC_SIZE; ++i) {
            newPositions[dropletId][i] = normalized[i];
        }
    }
    
    public static void calculateNewVelocities(float[][] newPositions, float[][] velocities,
            float[][] predictedVelocities, float[][] newVelocities, float[][] forces,
            Dpd.DropletParameter[] params, int[] types, int[] cells, int[] cellNeighbours, int step, int fullDropletId) {
        
        int dropletId = fullDropletId / numberOfCellNeighbours;
        if (dropletId >= numberOfDroplets) {
            return;
        }
    
        float localVel[][] = new float[numberOfCellNeighbours][VEC_SIZE];
        for(int dropletCellNeighbourId = 0; dropletCellNeighbourId < 27; ++dropletCellNeighbourId) {
            for(int i = 0; i < numberOfCellNeighbours; ++i) {
                float[] tmp = calculateForce(newPositions, velocities, params, types, cells, cellNeighbours, dropletId, dropletCellNeighbourId, step);
                for(int j = 0; j < VEC_SIZE; ++j) {
                    localVel[i][j] = tmp[j];
                }
            }
        }

        for(int i = 0; i < numberOfCellNeighbours; ++i) {
            for(int j = 0; j < VEC_SIZE; ++j) {
                newVelocities[dropletId][j] = velocities[dropletId][j] + 0.5f * deltaTime * (forces[dropletId][j] + localVel[i][j]);
            }
        }
    }

    public static int calculateHash(int d1, int d2) {
        int i1, i2;
        if (d1 <= d2) {
            i1 = d1;
            i2 = d2;
        } else {
            i1 = d2;
            i2 = d1;
        }
        return (i1 + i2) * (i1 + i2 + 1) / 2 + i2;
    }

    public static float rand(int seed, int step) {
        return new Random().nextFloat();
    }

    public static float normalRand(float U1, float U2) {
        float R = (float) (-2 * Math.log(U1));
        float fi = (float) (2 * Math.PI * U2);
        float Z1 = (float) (Math.sqrt(R) * Math.cos(fi));
        return Z1;
    }

    public static float gaussianRandom(int dropletId, int neighbourId, int step) {
        int seed = calculateHash(dropletId, neighbourId);
        float U1 = (float) ((rand(seed, step) + 1.0) / 2);
        float U2 = (float) ((rand(seed, step) + 1.0) / 2);
        return normalRand(U1, U2);
    }

    public static int calculateCellId(float[] position) {
        return ((int)((position[0] + boxSize) / cellRadius)) + 
                ((int)(2 * boxSize / cellRadius)) * (((int)((position[1] + initBoxWidth) / cellRadius)) + 
                ((int)(2 * initBoxWidth / cellRadius)) * ((int)((position[2] + boxSize) / cellRadius)));
    }

    private void printAverageVelocity() {
        float[] vel = new float[]{0f, 0f, 0f};
        for(float[] velocity : newVelocities) {
            for(int i = 0; i < VEC_SIZE; ++i) {
                vel[i] += velocity[i];
            }
        }        
        System.out.println("avgVel = (" + vel[0] / numberOfDroplets + ", " + vel[1] / numberOfDroplets + ", " + vel[2] / numberOfDroplets + ")");
    }
}
