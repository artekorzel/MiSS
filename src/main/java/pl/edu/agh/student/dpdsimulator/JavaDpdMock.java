package pl.edu.agh.student.dpdsimulator;

public class JavaDpdMock {

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

    public static float[] calculateConservativeForce(float[][] positions, float cutoffRadius,
                                                     float repulsionParameter, int numberOfDroplets, int dropletId) {

        float[] conservativeForce = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
        for (int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
            if (neighbourId != dropletId) {
                float dist = distance(positions[neighbourId], positions[dropletId]);
                if (dist < cutoffRadius) {
                    float factor = repulsionParameter * (1.0f - dist / cutoffRadius);
                    float[] normalized = normalize(positions[neighbourId], positions[dropletId]);
                    for (int k = 0; k < 4; ++k) {
                        conservativeForce[k] += factor * normalized[k];
                    }
                }
            }
        }
        return conservativeForce;
    }

    private static float[] normalize(float[] position, float[] position1) {
        float[] result = new float[4];
        float dist = distance(position, position1);
        for (int k = 0; k < 4; ++k) {
            result[k] = (position[k] - position1[k]) / dist;
        }
        return result;
    }

    private static float distance(float[] position, float[] position1) {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
            sum += (float) Math.pow(position[k] - position1[k], 2);
        }
        return (float) Math.sqrt(sum);
    }

    public static float[] calculateDissipativeForce(float[][] positions, float[][] velocities, float cutoffRadius, float gamma, int numberOfDroplets, int dropletId) {
        float[] dissipativeForce = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
        for (int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
            if (neighbourId != dropletId) {
                float dist = distance(positions[neighbourId], positions[dropletId]);
                if (dist < cutoffRadius) {
                    float weight = weightD(dist, cutoffRadius);
                    float[] normalizedVector = normalize(positions[neighbourId], positions[dropletId]);
                    float factor = gamma * weight * dot(normalizedVector, diff(velocities[neighbourId], velocities[dropletId]));
                    for (int k = 0; k < 4; ++k) {
                        dissipativeForce[k] -= factor * normalizedVector[k];
                    }
                }
            }
        }
        return dissipativeForce;
    }

    private static float dot(float[] normalizedVector, float[] diff) {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
            sum += normalizedVector[k] * diff[k];
        }
        return sum;
    }

    private static float[] diff(float[] velocity, float[] velocity1) {
        float[] result = new float[4];
        for (int k = 0; k < 4; ++k) {
            result[k] = velocity[k] - velocity1[k];
        }
        return result;
    }

    public static float[] calculateRandomForce(float[][] positions, float[] gaussianRandoms, float cutoffRadius, float sigma, int numberOfDroplets, int dropletId) {
        float[] randomForce = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
        for (int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
            if (neighbourId != dropletId) {
                float dist = distance(positions[neighbourId], positions[dropletId]);
                if (dist < cutoffRadius) {
                    float weight = weightR(dist, cutoffRadius);
                    float[] normalizedVector = normalize(positions[neighbourId], positions[dropletId]);
                    float factor = sigma * weight * gaussianRandoms[neighbourId * numberOfDroplets + dropletId];
                    for (int k = 0; k < 4; ++k) {
                        randomForce[k] += factor * normalizedVector[k];
                    }
                }
            }
        }
        return randomForce;
    }

    public static void initForces(float[][] positions, float[][] velocities, float[][] forces, float[] gaussianRandoms,
                                  float gamma, float sigma, float cutoffRadius, int numberOfDroplets, float repulsionParameter) {
        for (int dropletId = 0; dropletId < numberOfDroplets; ++dropletId) {
            float[] conservativeForce = calculateConservativeForce(positions, cutoffRadius, repulsionParameter, numberOfDroplets, dropletId);
            float[] dissipativeForce = calculateDissipativeForce(positions, velocities, cutoffRadius, gamma, numberOfDroplets, dropletId);
            float[] randomForce = calculateRandomForce(positions, gaussianRandoms, cutoffRadius, sigma, numberOfDroplets, dropletId);
            for (int k = 0; k < 4; ++k) {
                forces[dropletId][k] = conservativeForce[k] + dissipativeForce[k] + randomForce[k];
            }
        }
    }
}
