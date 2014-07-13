package pl.edu.agh.student.dpdsimulator;

public class JavaDpdMock {

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

    private static float[] fmod(float[] vector, float factor) {
        float[] result = new float[VEC_SIZE];
        for (int k = 0; k < VEC_SIZE; ++k) {
            result[k] = vector[k] - factor * ((int) (vector[k] / factor));
        }

        return result;
    }

    static float[] normalizePosition(float[] vector, float boxSize) {
        float[] vec = addConst(vector, boxSize);
        return addConst(fmod(addConst(fmod(vec, 2.0f * boxSize), 2.0f * boxSize), 2.0f * boxSize), -1 * boxSize);
    }

    private static float[] addConst(float[] vector, float boxSize) {
        float[] vec = new float[VEC_SIZE];
        for (int i = 0; i < VEC_SIZE; ++i) {
            vec[i] = vector[i] + boxSize;
        }
        return vec;
    }

    static float[] calculateConservativeForce(float[][] positions, float cutoffRadius,
            float repulsionParameter, int numberOfDroplets, int dropletId) {

        float[] conservativeForce = new float[]{0.0f, 0.0f, 0.0f};
        for (int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
            if (neighbourId != dropletId) {
                float dist = distance(positions[neighbourId], positions[dropletId]);
                if (dist < cutoffRadius) {
                    float[] normalized = normalize(positions[neighbourId], positions[dropletId]);
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        conservativeForce[i] += repulsionParameter * (1.0 - dist / cutoffRadius)
                                * normalized[i];
                    }
                }
            }
        }
        return conservativeForce;
    }

    static float[] calculateDissipativeForce(float[][] positions, float[][] velocities,
            float cutoffRadius, float gamma, int numberOfDroplets, int dropletId) {

        float[] dissipativeForce = new float[]{0.0f, 0.0f, 0.0f};
        for (int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
            if (neighbourId != dropletId) {
                float dist = distance(positions[neighbourId], positions[dropletId]);
                if (dist < cutoffRadius) {
                    float weight = weightD(dist, cutoffRadius);
                    float[] normalizedVector = normalize(positions[neighbourId], positions[dropletId]);
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        dissipativeForce[i] -= gamma * weight * dot(normalizedVector,
                                diff(velocities[neighbourId], velocities[dropletId])) * normalizedVector[i];
                    }
                }
            }
        }
        return dissipativeForce;
    }

    static float[] calculateRandomForce(float[][] positions, float[] gaussianRandoms,
            float cutoffRadius, float sigma, int numberOfDroplets, int dropletId) {

        float[] randomForce = new float[]{0.0f, 0.0f, 0.0f};
        for (int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
            if (neighbourId != dropletId) {
                float dist = distance(positions[neighbourId], positions[dropletId]);
                if (dist < cutoffRadius) {
                    float weight = weightR(dist, cutoffRadius);
                    float[] normalizedVector = normalize(positions[neighbourId], positions[dropletId]);
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        randomForce[i] += sigma * weight * gaussianRandoms[neighbourId
                                * numberOfDroplets + dropletId] * normalizedVector[i];
                    }
                }
            }
        }
        return randomForce;
    }

    static float[] calculateForce(float[][] positions, float[][] velocities, float[] gaussianRandoms,
            float gamma, float sigma, float cutoffRadius, int numberOfDroplets, float repulsionParameter, int dropletId) {

        float[] conservativeForce = calculateConservativeForce(positions,
                cutoffRadius, repulsionParameter, numberOfDroplets, dropletId);

        float[] dissipativeForce = calculateDissipativeForce(positions, velocities,
                cutoffRadius, gamma, numberOfDroplets, dropletId);

        float[] randomForce = calculateRandomForce(positions, gaussianRandoms,
                cutoffRadius, sigma, numberOfDroplets, dropletId);

        float[] result = new float[VEC_SIZE];
        for (int i = 0; i < VEC_SIZE; ++i) {
            result[i] = conservativeForce[i] + dissipativeForce[i] + randomForce[i];
        }
        return result;
    }

    public static void calculateForces(float[][] positions, float[][] velocities, float[][] forces,
            float[] gaussianRandoms, float gamma, float sigma,
            float cutoffRadius, int numberOfDroplets, float repulsionParameter, int dropletId) {

        float[] newForce = calculateForce(positions, velocities, gaussianRandoms, gamma, sigma, cutoffRadius,
                numberOfDroplets, repulsionParameter, dropletId);
        System.arraycopy(newForce, 0, forces[dropletId], 0, VEC_SIZE);
    }

    public static void calculateNewPositionsAndPredictedVelocities(float[][] positions, float[][] velocities,
            float[][] forces, float[][] newPositions, float[][] predictedVelocities,
            float deltaTime, float lambda, float boxSize, int dropletId) {

        float[] newPosition = new float[VEC_SIZE];
        for (int i = 0; i < VEC_SIZE; ++i) {
            newPosition[i] = positions[dropletId][i]
                    + deltaTime * velocities[dropletId][i] + 0.5f * deltaTime * deltaTime * forces[dropletId][i];
            predictedVelocities[dropletId][i] = velocities[dropletId][i] + lambda * deltaTime * forces[dropletId][i];
        }
        float[] normalized = normalizePosition(newPosition, boxSize);
        System.arraycopy(normalized, 0, newPositions[dropletId], 0, VEC_SIZE);
    }

    public static void calculateNewVelocities(float[][] newPositions, float[][] velocities,
            float[][] predictedVelocities, float[][] newVelocities, float[][] forces,
            float[] gaussianRandoms, float deltaTime, float gamma, float sigma,
            float cutoffRadius, int numberOfDroplets, float repulsionParameter, int dropletId) {

        float[] predictedForce = calculateForce(newPositions, predictedVelocities, gaussianRandoms, gamma, sigma,
                cutoffRadius, numberOfDroplets, repulsionParameter, dropletId);

        for (int i = 0; i < VEC_SIZE; ++i) {
            newVelocities[dropletId][i] = velocities[dropletId][i] + 0.5f * deltaTime * (forces[dropletId][i] + predictedForce[i]);
        }
    }
}
