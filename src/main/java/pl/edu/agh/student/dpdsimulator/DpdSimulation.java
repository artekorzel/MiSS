package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.util.ParallelRandom;
import org.bridj.Pointer;

public class DpdSimulation implements Simulation {

    public static final int numberOfDroplets = 10;
    public static final float timeDelta = 0.04f;
    public static final float cutoffRadius = 1.0f;
    public static final float boxSize = 10.0f;
    public static final float velocityInitRange = 1.0f;
    public static final float temperature = 293.1f;
    public static final float boltzmanConstant = (float) 1.3806488e-23;
    public static final float density = 3.0f;
    public static final float repulsionParameter = 75.0f * boltzmanConstant * temperature / density;
    public static final float sigma = 0.075f;
    public static final float gamma = sigma * sigma / 2 / boltzmanConstant / temperature;

    public void run() throws Exception {
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();

        float time = 0.0f;

        ParallelRandom random = new ParallelRandom();
        CLBuffer<float[]> positions = createVector(context, random, boxSize);
        CLBuffer<float[]> velocities = createVector(context, random, velocityInitRange);
        CLBuffer<float[]> forces = context.createBuffer(CLMem.Usage.InputOutput, float[].class, numberOfDroplets);

        CLBuffer<Float> gaussianRandoms = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * numberOfDroplets);

    }

    private CLBuffer<float[]> createVector(CLContext context, ParallelRandom random, float range) {
        Pointer<float[]> valuesPointer = Pointer.allocateArray(float[].class, numberOfDroplets);
        for(int i = 0; i < numberOfDroplets; ++i) {
            valuesPointer.set(i, new float[] {
                    nextRandomFloat(random, range),
                    nextRandomFloat(random, range),
                    nextRandomFloat(random, range)
            });
        }
        return context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }

    private float nextRandomFloat(ParallelRandom random, float range) {
        return random.nextFloat() * 2 * range - range;
    }

//    private
}
