package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import static pl.edu.agh.student.dpdsimulator.StartParameters.*;

public class DpdSimulation implements Simulation {

    public static final int VECTOR_SIZE = 4;
    public static final int NUMBER_OF_REDUCTION_KERNELS = 5;
    private Random random;
    private CLBuffer<Float> positions;
    private CLBuffer<Float> newPositions;
    private CLBuffer<Float> velocities;
    private CLBuffer<Float> predictedVelocities;
    private CLBuffer<Float> newVelocities;
    private CLBuffer<Float> forces; 
    private CLBuffer<Float> partialSums;
    private CLBuffer<Float> output;
    
    private CLContext context;
    private CLQueue queue;
    private Dpd dpdKernel;
    private int[] globalSizes;
    private int[] localSizes;
    int step;

    @Override
    public void run() throws Exception {
        initData();
        performSimulation();
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
        partialSums = context.createFloatBuffer(CLMem.Usage.InputOutput, NUMBER_OF_REDUCTION_KERNELS * VECTOR_SIZE);
        output = context.createFloatBuffer(CLMem.Usage.InputOutput, VECTOR_SIZE);
        
        dpdKernel = new Dpd(context);
        globalSizes = new int[]{numberOfDroplets};
        localSizes = new int[]{NUMBER_OF_REDUCTION_KERNELS};
    }

    private CLBuffer<Float> createVector(float range) {
        long numberOfCoordinates = numberOfDroplets * VECTOR_SIZE;
        Pointer<Float> valuesPointer = allocateFloats(numberOfCoordinates).order(context.getByteOrder());
        for (int i = 0; i < numberOfCoordinates; i += VECTOR_SIZE) {
            valuesPointer.set(i, nextRandomFloat(range));
            valuesPointer.set(i + 1, nextRandomFloat(range));
            valuesPointer.set(i + 2, nextRandomFloat(range));
            valuesPointer.set(i + 3, 0.0f);
        }
        CLBuffer<Float> buffer = context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
        valuesPointer.release();
        return buffer;
    }

    private float nextRandomFloat(float range) {
        return random.nextFloat() * 2 * range - range;
    }

    private Pointer<Float> allocateFloats(long size) {
        return Pointer.allocateFloats(size).order(context.getByteOrder());
    }

    private void performSimulation() {
//        printVectors("\nPositions", "pos", queue, positions);
//        printVectors("\nVelocities", "vel", queue, velocities);

        CLEvent loopEndEvent = null;
        for (step = 1; step <= numberOfSteps; ++step) {
            System.out.println("Step: " + step);
            
            loopEndEvent = performSingleStep(loopEndEvent);

            writePositionsFile(newPositions, loopEndEvent);
//            printVectors("\nPositions", "pos", queue, newPositions, loopEndEvent);
//            printVectors("\nVelocities", "vel", queue, newVelocities, loopEndEvent);
//            printVectors("\nForces", "force", queue, forces, loopEndEvent);
//            writeAvg(output ,loopEndEvent);
            swapPositions();
            swapVelocities();
        }
    }

    private CLEvent performSingleStep(CLEvent previousStepEvent) {
        CLEvent forcesEvent = calculateForces(dpdKernel, globalSizes, previousStepEvent);
        CLEvent newPositionsAndPredictedVelocitiesEvent = 
                calculateNewPositionsAndPredictedVelocities(dpdKernel, globalSizes, forcesEvent);
        return calculateNewVelocities(dpdKernel, globalSizes, newPositionsAndPredictedVelocitiesEvent);
    }

    private CLEvent calculateForces(Dpd dpdKernel, int[] globalSizes, CLEvent gaussianRandomsEvent) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, gamma, sigma, cutoffRadius, 
                repulsionParameter, numberOfDroplets, step, globalSizes, null, gaussianRandomsEvent);
    }

    private CLEvent calculateNewPositionsAndPredictedVelocities(Dpd dpdKernel, int[] globalSizes, CLEvent forcesEvent) {
        return dpdKernel.calculateNewPositionsAndPredictedVelocities(queue, positions, velocities, forces, newPositions,
                predictedVelocities, deltaTime, lambda, numberOfDroplets, boxSize, globalSizes, null, forcesEvent);
    }

    private CLEvent calculateNewVelocities(Dpd dpdKernel, int[] globalSizes, CLEvent newPositionsAndPredictedVelocitiesEvent) {
        return dpdKernel.calculateNewVelocities(queue, newPositions, velocities, predictedVelocities, newVelocities,
                forces, deltaTime, gamma, sigma, cutoffRadius, repulsionParameter, numberOfDroplets, step, 
                globalSizes, null, newPositionsAndPredictedVelocitiesEvent);
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

    private void writePositionsFile(CLBuffer<Float> buffer, CLEvent... events) {
        try (FileWriter writer = new FileWriter("../results/result" + step + ".csv")) {
            Pointer<Float> out = buffer.read(queue, events);
            writer.write("x, y, z\n");
            for (int i = 0; i < numberOfDroplets; i++) {
                writer.write(out.get(i * VECTOR_SIZE) + ","
                        + out.get(i * VECTOR_SIZE + 1) + ","
                        + out.get(i * VECTOR_SIZE + 2) + "\n");
            }
            out.release();
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
        out.release();
    }

    private void writeAvg(CLBuffer<Float> buffer, CLEvent loopEndEvent) {
        CLEvent reductionEvent = dpdKernel.reductionVector(queue, newVelocities, partialSums, 
                output, numberOfDroplets, localSizes, localSizes, loopEndEvent);
        Pointer<Float> out = buffer.read(queue, reductionEvent);
        System.out.println("avgVel = (" + out.get(0) + ", " + out.get(1) + ", " + out.get(2) + ")");
        out.release();
    }        
}


