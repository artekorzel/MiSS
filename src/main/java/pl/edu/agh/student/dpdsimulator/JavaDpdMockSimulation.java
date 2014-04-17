package pl.edu.agh.student.dpdsimulator;

import java.io.IOException;
import java.util.Random;

import static pl.edu.agh.student.dpdsimulator.StartParameters.*;

public class JavaDpdMockSimulation implements Simulation {

    private Random random;
    private float[][] positions;
    private float[][] newPositions;
    private float[][] velocities;
    private float[][] predictedVelocities;
    private float[][] newVelocities;
    private float[][] forces;
    private float[] gaussianRandoms;

    public void run() throws Exception {
        initData();

        printVectors("\nBefore calculations", "pos", positions);
        printVectors("\nVelocities", "vel", velocities);

        performSimulation();

        printVectors("\nForces", "force", forces);
        printVectors("\nPositions", "pos", newPositions);
        printVectors("\nVelocities", "vel", newVelocities);
    }

    private void initData() throws IOException {
        random = new Random();
        positions = createVector(boxSize);
        newPositions = new float[numberOfDroplets][JavaDpdMock.VEC_SIZE];
        velocities = createVector(velocityInitRange);
        predictedVelocities = new float[numberOfDroplets][JavaDpdMock.VEC_SIZE];
        newVelocities = new float[numberOfDroplets][JavaDpdMock.VEC_SIZE];
        forces = new float[numberOfDroplets][JavaDpdMock.VEC_SIZE];
        gaussianRandoms = new float[numberOfDroplets * numberOfDroplets];
    }

    private float[][] createVector(float range) {
        float[][] generatedValues = new float[numberOfDroplets][JavaDpdMock.VEC_SIZE];
        for (int i = 0; i < numberOfDroplets; ++i) {
            generatedValues[i][0] = nextRandomFloat(range);
            generatedValues[i][1] = nextRandomFloat(range);
            generatedValues[i][2] = nextRandomFloat(range);
        }
        return generatedValues;
    }

    private float nextRandomFloat(float range) {
        return random.nextFloat() * 2 * range - range;
    }

    private void performSimulation() {
        for (int i = 0, n = numberOfSteps - 1; i < n; ++i) {
            performSingleStep();


            printVectors("\nPositions", "pos", newPositions);
            printVectors("\nVelocities", "vel", newVelocities);
            printVectors("\nForces", "force", forces);

            swapPositions();
            swapVelocities();
        }
        performSingleStep();
    }

    private void performSingleStep() {
        for(int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            writeGaussianRandoms();
            calculateForces(dropletId);
            calculateNewPositionsAndPredictedVelocities(dropletId);
            calculateNewVelocities(dropletId);
        }
    }

    private void writeGaussianRandoms() {
        for (int i = 0; i < numberOfDroplets; ++i) {
            for (int j = i; j < numberOfDroplets; ++j) {
                float randomValue = (float) random.nextGaussian();
                gaussianRandoms[i * numberOfDroplets + j] = randomValue;
                gaussianRandoms[j * numberOfDroplets + i] = randomValue;
            }
        }
    }

    private void calculateForces(int dropletId) {
        JavaDpdMock.calculateForces(positions, velocities, forces, gaussianRandoms,
                gamma, sigma, cutoffRadius, numberOfDroplets, repulsionParameter, dropletId);
    }

    private void calculateNewPositionsAndPredictedVelocities(int dropletId) {
        JavaDpdMock.calculateNewPositionsAndPredictedVelocities(positions, velocities, forces, newPositions,
                predictedVelocities, timeDelta, lambda, boxSize, dropletId);
    }

    private void calculateNewVelocities(int dropletId) {
        JavaDpdMock.calculateNewVelocities(newPositions, velocities, predictedVelocities,
                newVelocities, forces, gaussianRandoms, timeDelta, gamma, sigma, cutoffRadius, numberOfDroplets,
                repulsionParameter, dropletId);
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

    private void printVectors(String intro, String name, float[][] data) {
        System.out.println(intro);
        for (int i = 0; i < numberOfDroplets; i++) {
            System.out.println(name + "[" + i + "] = ("
                    + data[i][0] + ", "
                    + data[i][1] + ", "
                    + data[i][2] + ")");
        }
    }
}


