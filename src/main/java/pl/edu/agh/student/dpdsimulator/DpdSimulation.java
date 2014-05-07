package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import java.awt.BorderLayout;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import static pl.edu.agh.student.dpdsimulator.StartParameters.*;

public class DpdSimulation implements Simulation {

    public static final int VECTOR_SIZE = 3;
    public static final int NUMBER_OF_REDUCTION_KERNELS = 5;
    private Random random;
    private CLBuffer<Float> positions;
    private CLBuffer<Float> newPositions;
    private CLBuffer<Float> velocities;
    private CLBuffer<Float> predictedVelocities;
    private CLBuffer<Float> newVelocities;
    private CLBuffer<Float> forces;
    private CLBuffer<Float> gaussianRandoms;    
    private CLBuffer<Float> partialSums;
    private CLBuffer<Float> output;
    
    private Pointer<Float> gaussianRandomsData;
//    private float time;
    private CLContext context;
    private CLQueue queue;
    private Dpd dpdKernel;
    private int[] globalSizes;
    private int[] localSizes;
    private int[] globalSizes2;

    public void run() throws Exception {
        initData();

//        printVectors("\nPositions", "pos", queue, positions);
//        printVectors("\nVelocities", "vel", queue, velocities);

        CLEvent simulationEndEvent = performSimulation();

//        printVectors("\nPositions", "pos", queue, newPositions, simulationEndEvent);
//        printVectors("\nVelocities", "vel", queue, newVelocities, simulationEndEvent);
//        printVectors("\nForces", "force", queue, forces, simulationEndEvent);
    }

    private void initData() throws IOException {
        context = JavaCL.createBestContext();
        queue = context.createDefaultQueue();

        random = new Random();
        positions = createVector(boxSize);
        newPositions = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        velocities = createVector(velocityInitRange);
        predictedVelocities = context.createFloatBuffer(CLMem.Usage.Input, numberOfDroplets * VECTOR_SIZE);
        newVelocities = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        forces = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        gaussianRandoms = context.createFloatBuffer(CLMem.Usage.Input, numberOfDroplets * numberOfDroplets);
        gaussianRandomsData = allocateFloats(numberOfDroplets * numberOfDroplets);
        partialSums = context.createFloatBuffer(CLMem.Usage.InputOutput, NUMBER_OF_REDUCTION_KERNELS*VECTOR_SIZE);
        output = context.createFloatBuffer(CLMem.Usage.InputOutput, VECTOR_SIZE);
        
        dpdKernel = new Dpd(context);
        globalSizes = new int[]{numberOfDroplets};
        localSizes = new int[]{NUMBER_OF_REDUCTION_KERNELS};
//        globalSizes2 = new int[]{1};
//        time = 0.0f;
    }

    private CLBuffer<Float> createVector(float range) {
        int numberOfCoordinates = numberOfDroplets * VECTOR_SIZE;
        Pointer<Float> valuesPointer = allocateFloats(numberOfCoordinates);
        for (int i = 0; i < numberOfCoordinates; i += VECTOR_SIZE) {
            valuesPointer.set(i, nextRandomFloat(range));
            valuesPointer.set(i + 1, nextRandomFloat(range));
            valuesPointer.set(i + 2, nextRandomFloat(range));
//            for(int j = 3; j < VECTOR_SIZE; ++j) {
//                valuesPointer.set(i + j, 0.0f);
//            }
        }
        return context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }

    private float nextRandomFloat(float range) {
        return random.nextFloat() * 2 * range - range;
    }

    private Pointer<Float> allocateFloats(int size) {
        return Pointer.allocateFloats(size).order(context.getByteOrder());
    }

    private CLEvent performSimulation() {
        CLEvent loopEndEvent = null;      
        for (int i = 0, n = numberOfSteps - 1; i < n; ++i) {
            loopEndEvent = performSingleStep(dpdKernel, globalSizes, loopEndEvent);

            writePositionsFile(i, queue, newPositions, loopEndEvent);
            printVectors("\nPositions", "pos", queue, positions);
//            printVectors("\nVelocities", "vel", queue, newVelocities, loopEndEvent);
//            printVectors("\nForces", "force", queue, forces, loopEndEvent);
            writeAvg(queue, output ,loopEndEvent);
            swapPositions();
            swapVelocities();
        }
        loopEndEvent = performSingleStep(dpdKernel, globalSizes, loopEndEvent);
        return loopEndEvent;
    }

    private CLEvent performSingleStep(Dpd dpdKernel, int[] globalSizes, CLEvent previousStepEvent) {
        CLEvent gaussianRandomsEvent = writeGaussianRandoms(previousStepEvent);
        CLEvent forcesEvent = calculateForces(dpdKernel, globalSizes, gaussianRandomsEvent);
        CLEvent newPositionsAndPredictedVelocitiesEvent = calculateNewPositionsAndPredictedVelocities(dpdKernel, globalSizes, forcesEvent);
        return calculateNewVelocities(dpdKernel, globalSizes, newPositionsAndPredictedVelocitiesEvent);
    }

    private CLEvent writeGaussianRandoms(CLEvent previousStepEvent) {
        for (int i = 0; i < numberOfDroplets; ++i) {
            for (int j = i; j < numberOfDroplets; ++j) {
                float randomValue = (float) random.nextGaussian();
                gaussianRandomsData.set(i * numberOfDroplets + j, randomValue);
                gaussianRandomsData.set(j * numberOfDroplets + i, randomValue);
            }
        }
        return gaussianRandoms.write(queue, gaussianRandomsData, true, previousStepEvent);
    }

    private CLEvent calculateForces(Dpd dpdKernel, int[] globalSizes, CLEvent gaussianRandomsEvent) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, gaussianRandoms, gamma, sigma,
                cutoffRadius, repulsionParameter, numberOfDroplets, globalSizes, null, gaussianRandomsEvent);
    }

    private CLEvent calculateNewPositionsAndPredictedVelocities(Dpd dpdKernel, int[] globalSizes, CLEvent forcesEvent) {
        return dpdKernel.calculateNewPositionsAndPredictedVelocities(queue, positions, velocities, forces, newPositions,
                predictedVelocities, deltaTime, lambda, numberOfDroplets, boxSize, globalSizes, null, forcesEvent);
    }

    private CLEvent calculateNewVelocities(Dpd dpdKernel, int[] globalSizes, CLEvent newPositionsAndPredictedVelocitiesEvent) {
        return dpdKernel.calculateNewVelocities(queue, newPositions, velocities, predictedVelocities, newVelocities,
                forces, gaussianRandoms, deltaTime, gamma, sigma, cutoffRadius, repulsionParameter,
                numberOfDroplets, globalSizes, null, newPositionsAndPredictedVelocitiesEvent);
    }

    private void swapPositions() {
        CLBuffer<Float> tmp = positions;
        positions = newPositions;
        newPositions = tmp;
    }

    private void swapVelocities() {
        CLBuffer<Float> tmp = velocities;
        velocities = newVelocities;
        newVelocities = tmp;
    }

    private void writePositionsFile(int step, CLQueue queue, CLBuffer<Float> buffer, CLEvent... events) {
        try (FileWriter writer = new FileWriter("../result" + step + ".csv")) {
            Pointer<Float> out = buffer.read(queue, events);
            writer.write("x, y, z\n");
            for (int i = 0; i < numberOfDroplets; i++) {
                writer.write(out.get(i * VECTOR_SIZE) + ","
                        + out.get(i * VECTOR_SIZE + 1) + ","
                        + out.get(i * VECTOR_SIZE + 2) + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void printVectors(String intro, String name, CLQueue queue, CLBuffer<Float> buffer, CLEvent... events) {
        System.out.println(intro);
        Pointer<Float> out = buffer.read(queue, events);
        for (int i = 0; i < numberOfDroplets; i++) {
            System.out.println(name + "[" + i + "] = ("
                    + out.get(i * VECTOR_SIZE) + ", "
                    + out.get(i * VECTOR_SIZE + 1) + ", "
                    + out.get(i * VECTOR_SIZE + 2) + ")");
        }
    }

    private void writeAvg(CLQueue queue, CLBuffer<Float> buffer, CLEvent loopEndEvent) {
        CLEvent reductionEvent = dpdKernel.reductionVector(queue, newPositions, partialSums, output, numberOfDroplets, globalSizes, localSizes, loopEndEvent);
        Pointer<Float> out = buffer.read(queue, reductionEvent);
        System.out.println("avgVel = (" + out.get(0) + ", " + out.get(1) + ", " + out.get(2) + ")");
        Pointer<Float> out2 = partialSums.read(queue, reductionEvent);
        System.out.println("Partials:");
        for(int i = 0; i < NUMBER_OF_REDUCTION_KERNELS; i++){           
            System.out.println("(" + out2.get(0) + ", " + out.get(1) + ", " + out.get(2) + ")");
        }
    }
}


