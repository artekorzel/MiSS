package pl.edu.agh.student.dpdsimulator;

import java.util.Random;

import static pl.edu.agh.student.dpdsimulator.StartParameters.*;

public class DpdMockSimulation implements Simulation {

    public void run() throws Exception {
        Random random = new Random();

        float[][] positions = createTestVector(random, boxSize);
        float[][] velocities = createTestVector(random, velocityInitRange);
        float[][] forces = createTestVector(random, 1.0f);

        float[] gaussianRandoms = new float[numberOfDroplets * numberOfDroplets];
        for (int i = 0; i < numberOfDroplets; ++i) {
            for (int j = i; j < numberOfDroplets; ++j) {
                float randomValue = (float) random.nextGaussian();
                gaussianRandoms[i * numberOfDroplets + j] = randomValue;
                gaussianRandoms[j * numberOfDroplets + i] = randomValue;
            }
        }
        JavaDpdMock.initForces(positions, velocities, forces, gaussianRandoms, gamma, sigma, cutoffRadius, numberOfDroplets, repulsionParameter);

        for (int i = 0; i < numberOfDroplets; i++) {
            System.out.print("forces[" + i + "] = (");
            for (int j = 0; j < 3; j++) {
                System.out.print(forces[i][j] + ", ");
            }
            System.out.println(forces[i][3] + ")");
        }
    }

    private float[][] createTestVector(Random random, float range) {
        float[][] result = new float[numberOfDroplets][4];
        for (int i = 0; i < numberOfDroplets; i++) {
            result[i][0] = nextRandomFloat(random, range);
            result[i][1] = nextRandomFloat(random, range);
            result[i][2] = nextRandomFloat(random, range);
            result[i][3] = 0.0f;
        }
        return result;
    }

    private float nextRandomFloat(Random random, float range) {
        return random.nextFloat() * 2 * range - range;
    }
}
