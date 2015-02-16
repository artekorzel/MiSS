package pl.edu.agh.student.dpdsimulator;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
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
    private int[] states;
    private Dpd.Parameters[] parameters;
    private int step;

    @Override
    public void initData(float boxSizeScale, float boxWidthScale, int numberOfDropletsParam) throws IOException {
        super.initData(boxSizeScale, boxWidthScale, numberOfDropletsParam);
        random = new Random();
        cells = new int[maxDropletsPerCell * numberOfCells];
        cellNeighbours = new int[numberOfCells * numberOfCellNeighbours];
        positions = new float[numberOfDroplets][VEC_SIZE];
        newPositions = new float[numberOfDroplets][VEC_SIZE];
        velocities = new float[numberOfDroplets][VEC_SIZE];
        predictedVelocities = new float[numberOfDroplets][VEC_SIZE];
        newVelocities = new float[numberOfDroplets][VEC_SIZE];
        forces = new float[numberOfDroplets][VEC_SIZE];
        states = new int[numberOfDroplets];
    }    
    
    @Override
    public void performSimulation() {        
        long startTime = System.nanoTime();
        step = 0;
        initSimulationData();
        long endInitTime = System.nanoTime();
        for (step = 1; step <= numberOfSteps; ++step) {
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
        List<Dpd.Parameters> params = super.createParameters();
        parameters = params.toArray(new Dpd.Parameters[params.size()]);
    }

    private void initStates(){
        for(int i = 0; i < numberOfDroplets; i++){
            states[i] = random.nextInt(Integer.MAX_VALUE);
        }
    }
    
    private void initPositionsAndVelocities() {
        generateTube(positions, states, numberOfDroplets, boxSize, boxWidth);
        generateRandomVector(velocities, states, numberOfDroplets);
    }
    
    void generateTube(float[][] vector, int[] states, int numberOfDroplets, 
            float boxSize, float boxWidth) {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            int seed = states[dropletId];   

            float x = (rand(seed, 1) * 2 - 1) * boxSize;
            float y = (rand(seed, 1) * 2 - 1) * boxWidth;
            float z = (rand(seed, 1) * 2 - 1) * boxSize;

            states[dropletId] = seed;
            vector[dropletId][0] = x;
            vector[dropletId][1] = y;
            vector[dropletId][2] = z;
        }
    }
    
    void generateRandomVector(float[][] vector, int[] states, int numberOfDroplets) {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            float x, y, z;
            int seed = states[dropletId];
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
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
            calculateForces(positions, velocities, forces, parameters, cells, cellNeighbours, step, dropletId);
        }
    }

    private void calculateNewPositionsAndPredictedVelocities() {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            calculateNewPositionsAndPredictedVelocities(positions, velocities, forces, newPositions,
                    predictedVelocities, dropletId);
        }
    }

    private void calculateNewVelocities() {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            calculateNewVelocities(newPositions, velocities, predictedVelocities, newVelocities, forces, parameters, 
                    cells, cellNeighbours, step, dropletId);
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

    private static float[] fmod(float[] vector, float[] factor) {
        float[] result = new float[VEC_SIZE];
        for (int k = 0; k < VEC_SIZE; ++k) {
            result[k] = vector[k] - factor[k] * ((int) (vector[k] / factor[k]));
        }

        return result;
    }

    static float[] normalizePosition(float[] vector) {
        float[] factor = new float[]{boxSize, boxWidth, boxSize};
        float[] factor1 = new float[]{2f * boxSize, 2f * boxWidth, 2f * boxSize};
        float[] factor2 = new float[]{-boxSize, -boxWidth, -boxSize};
        return add(fmod(add(fmod(add(vector, factor), factor1), factor1), factor1), factor2);
    }

    private static float[] add(float[] vector, float[] vector2) {
        float[] vec = new float[VEC_SIZE];
        for (int i = 0; i < VEC_SIZE; ++i) {
            vec[i] = vector[i] + vector2[i];
        }
        return vec;
    }
    
    public static int calculateCellId(float[] position) {
        return ((int)((position[0] + boxSize) / cellRadius)) + 
                ((int)(2 * boxSize / cellRadius)) * (((int)((position[1] + boxWidth) / cellRadius)) + 
                        ((int)(2 * boxWidth / cellRadius)) * ((int)((position[2] + boxSize) / cellRadius)));
    }

    public static int[] calculateCellCoordinates(int dropletCellId, float cellRadius, float boxSize, 
            float boxWidth, int cellsNoXZ, int cellsNoY) {
        int dropletCellZ = dropletCellId / (cellsNoXZ * cellsNoY);
        int dropletCellX = dropletCellId - dropletCellZ * cellsNoXZ * cellsNoY;
        int dropletCellY = dropletCellX / cellsNoXZ;
        dropletCellX = dropletCellId % cellsNoXZ;
        return new int[] {dropletCellX, dropletCellY, dropletCellZ};
    }

    public static float[] getNeighbourPosition(float[][] positions, int[] dropletCellCoordinates, 
            float cellRadius, int dropletCellId, int neighbourCellId, int neighbourId, 
            float boxSize, float boxWidth, int cellsNoXZ, int cellsNoY) {
        float x = positions[neighbourId][0];
        float y = positions[neighbourId][1];
        float z = positions[neighbourId][2];
        float[] position = new float[] {x,y,z};

        if(neighbourCellId != dropletCellId) {
            int[] neighbourCellCoordinates = calculateCellCoordinates(neighbourCellId, cellRadius, boxSize, boxWidth, cellsNoXZ, cellsNoY);

            if(dropletCellCoordinates[0] == 0 && neighbourCellCoordinates[0] == cellsNoXZ - 1) {
                position[0] -= boxSize;
            }

            if(dropletCellCoordinates[0] == cellsNoXZ - 1 && neighbourCellCoordinates[0] == 0) {
                position[0] += boxSize;
            }

            if(dropletCellCoordinates[1] == 0 && neighbourCellCoordinates[1] == cellsNoY - 1) {
                position[1] -= boxWidth;
            }

            if(dropletCellCoordinates[1] == cellsNoY - 1 && neighbourCellCoordinates[1] == 0) {
                position[1] += boxWidth;
            }

            if(dropletCellCoordinates[2] == 0 && neighbourCellCoordinates[2] == cellsNoXZ - 1) {
                position[2] -= boxSize;
            }

            if(dropletCellCoordinates[2] == cellsNoXZ - 1 && neighbourCellCoordinates[2] == 0) {
                position[2] += boxSize;
            }
        }

        return position;
    }
    
    public static float[] calculateForce(float[][] positions, float[][] velocities, Dpd.Parameters[] parameters,
        int[] cells, int[] cellNeighbours, int dropletId, int dropletCellNeighbourId, int step) {

        float[] conservativeForce = new float[]{0.0f, 0.0f, 0.0f};

        float[] dropletPosition = positions[dropletId];
        float repulsionParameter = parameters[0].pi();
        float cutoffRadius = parameters[0].cutoffRadius();

        int dropletCellId = calculateCellId(dropletPosition);
        int cellsNoXZ = (int)(2 * boxSize / cellRadius);
        int cellsNoY = (int)(2 * boxWidth / cellRadius);
        int[] dropletCellCoordinates = calculateCellCoordinates(dropletCellId, cellRadius, boxSize, boxWidth, cellsNoXZ, cellsNoY);

        int cellId = cellNeighbours[dropletCellId * numberOfCellNeighbours + dropletCellNeighbourId];
        int neighbourId = 0, j;
        for(j = 0, neighbourId = cells[maxDropletsPerCell * cellId]; neighbourId >= 0; neighbourId = cells[maxDropletsPerCell * cellId + ++j]) {
            if(neighbourId != dropletId) {
                float[] neighbourPosition = getNeighbourPosition(positions, dropletCellCoordinates, 
                    cellRadius, dropletCellId, cellId, neighbourId, boxSize, boxWidth, cellsNoXZ, cellsNoY);
                float distanceValue = distance(neighbourPosition, dropletPosition);

                if(distanceValue < cutoffRadius) {
                    
                    float[] normalizedPositionVector = normalize(neighbourPosition, dropletPosition);

                    for(int k = 0; k < VEC_SIZE; ++k) {
                        conservativeForce[k] += repulsionParameter
                                * (1.0f - distanceValue / cutoffRadius) * normalizedPositionVector[k];
                    }
                }
            }
        }

        return conservativeForce;
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

    void fillCellNeighbours(int[] cellNeighbours) {

        for(int cellId = 0; cellId < numberOfCells; ++cellId) {

        int numberOfCellsPerXZDim = (int) Math.ceil(2 * boxSize / cellRadius);
        int numberOfCellsPerYDim = (int) Math.ceil(2 * boxWidth / cellRadius);
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
    }
        
    public static void calculateForces(float[][] positions, float[][] velocities, float[][] forces,
            Dpd.Parameters[] params, int[] cells, int[] cellNeighbours, int step, int fullDropletId) {
        
        int dropletId = fullDropletId / numberOfCellNeighbours;
        if (dropletId >= numberOfDroplets) {
            return;
        }
    
        float localForces[][] = new float[numberOfCellNeighbours][VEC_SIZE];
        for(int dropletCellNeighbourId = 0; dropletCellNeighbourId < numberOfCellNeighbours; ++dropletCellNeighbourId) {
            float[] tmp = calculateForce(positions, velocities, params, 
                    cells, cellNeighbours, dropletId, dropletCellNeighbourId, step);
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
            newPosition[i] = positions[dropletId][i] + deltaTime * velocities[dropletId][i];
        }
        float[] normalized = normalizePosition(newPosition);
        for (int i = 0; i < VEC_SIZE; ++i) {
            newPositions[dropletId][i] = normalized[i];
        }
    }
    
    public static void calculateNewVelocities(float[][] newPositions, float[][] velocities,
            float[][] predictedVelocities, float[][] newVelocities, float[][] forces,
            Dpd.Parameters[] params, int[] cells, int[] cellNeighbours, int step, int dropletId) {
        
        for(int j = 0; j < VEC_SIZE; ++j) {
            newVelocities[dropletId][j] = velocities[dropletId][j] + deltaTime * forces[dropletId][j] / params[0].mass();
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
    
    float calculateVelocitiesEnergy(float[][] velocities, int numberOfDroplets) {
        float energy = 0f;
        for(int globalId = 0; globalId < numberOfDroplets; ++globalId) {
            float[] velocity = velocities[globalId];
            energy += velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2];
        }
        return energy;
    }
    
    private void printAverageVelocity() {
        System.out.println(String.format(Locale.GERMANY, "%e", calculateVelocitiesEnergy(newVelocities, numberOfDroplets)));
    }
}
