package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

import java.util.Random;

import static pl.edu.agh.student.dpdsimulator.StartParameters.*;

public class DpdSimulation implements Simulation {

    public void run() throws Exception {
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();

        float time = 0.0f;

        Random random = new Random();
        CLBuffer<Float> positions = createVector(context, random, boxSize);
        CLBuffer<Float> newPositions = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * 4);
        CLBuffer<Float> velocities = createVector(context, random, velocityInitRange);
        CLBuffer<Float> predictedVelocities = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * 4);
        CLBuffer<Float> newVelocities = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * 4);
        CLBuffer<Float> forces = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * 4);

        Pointer<Float> gaussianRandomsData = Pointer.allocateFloats(numberOfDroplets * numberOfDroplets);
        for (int i = 0; i < numberOfDroplets; ++i) {
            for (int j = i; j < numberOfDroplets; ++j) {
                float randomValue = (float) random.nextGaussian();
                gaussianRandomsData.set(i * numberOfDroplets + j, randomValue);
                gaussianRandomsData.set(j * numberOfDroplets + i, randomValue);
            }
        }
        CLBuffer<Float> gaussianRandoms = context.createFloatBuffer(CLMem.Usage.Input, gaussianRandomsData);

        printVectors("\nBefore calculations", "pos", queue, positions);

        Dpd dpdKernel = new Dpd(context);
        int[] globalSizes = new int[]{numberOfDroplets};

        CLEvent forcesEvent = dpdKernel.calculateForces(queue, positions, velocities, forces, gaussianRandoms, time,
                timeDelta, lambda, gamma, sigma, cutoffRadius, numberOfDroplets, repulsionParameter, globalSizes, null);

        CLEvent newPositionsAndPredictedVelocitiesEvent =
                dpdKernel.calculateNewPositionsAndPredictedVelocities(queue, positions, velocities, forces, newPositions,
                        predictedVelocities, timeDelta, lambda, numberOfDroplets, boxSize, globalSizes, null, forcesEvent);

        CLEvent newVelocitiesEvent = dpdKernel.calculateNewVelocities(queue, newPositions, velocities, predictedVelocities,
                newVelocities, forces, gaussianRandoms, timeDelta, lambda, gamma, sigma, cutoffRadius, numberOfDroplets,
                repulsionParameter, globalSizes, null, newPositionsAndPredictedVelocitiesEvent);

        printVectors("\nPositions", "pos", queue, newPositions, newVelocitiesEvent);
        printVectors("\nVelocities", "vel", queue, newVelocities, newVelocitiesEvent);
    }

    private void printVectors(String intro, String name, CLQueue queue, CLBuffer<Float> buffer, CLEvent... events) {
        System.out.println(intro);
        Pointer<Float> out = buffer.read(queue, events);
        for (int i = 0; i < numberOfDroplets; i++) {
            System.out.print(name + "[" + i + "] = (");
            for (int j = 0; j < 3; j++) {
                System.out.print(out.get(i * 4 + j) + ", ");
            }
            System.out.println(out.get(i * 4 + 3) + ")");
        }
    }

    private CLBuffer<Float> createVector(CLContext context, Random random, float range) {
        int numberOfCoordinates = numberOfDroplets * 4;
        Pointer<Float> valuesPointer = Pointer.allocateArray(Float.class, numberOfCoordinates);
        for (int i = 0; i < numberOfCoordinates; i += 4) {
            valuesPointer.set(i, nextRandomFloat(random, range));
            valuesPointer.set(i + 1, nextRandomFloat(random, range));
            valuesPointer.set(i + 2, nextRandomFloat(random, range));
            valuesPointer.set(i + 3, 0.0f);
        }
        return context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }

    private float nextRandomFloat(Random random, float range) {
        return random.nextFloat() * 2 * range - range;
    }

}


