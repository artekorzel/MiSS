package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import org.bridj.Pointer;

import static pl.edu.agh.student.dpdsimulator.StartParameters.*;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd.DropletParameter;


public class DpdSimulation implements Simulation {

    public static final int VECTOR_SIZE = 4;
    private Random random;
    private CLBuffer<Float> positions;
    private CLBuffer<Float> newPositions;
    private CLBuffer<Float> velocities;
    private CLBuffer<Float> predictedVelocities;
    private CLBuffer<Float> newVelocities;
    private CLBuffer<Float> forces;
    private CLBuffer<Float> partialSums;
    private CLBuffer<Float> averageVelocity;
    private CLBuffer<Integer> types;
    private CLBuffer<DropletParameter> dropletParameters;
    private CLContext context;
    private CLQueue queue;
    private Dpd dpdKernel;
    private int[] globalSizes;
    private int[] localSizes;
    int step;
    int initialRandom;

    @Override
    public void run() throws Exception {
        initData();
        performSimulation();
    }

    private void initData() throws IOException {
        context = JavaCL.createBestContext();
        queue = context.createDefaultQueue();

        random = new Random();       
        positions = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        newPositions = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        velocities = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        predictedVelocities = context.createFloatBuffer(CLMem.Usage.Input, numberOfDroplets * VECTOR_SIZE);
        newVelocities = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        forces = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        partialSums = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets * VECTOR_SIZE);
        averageVelocity = context.createFloatBuffer(CLMem.Usage.InputOutput, VECTOR_SIZE);
        types = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, DropletParameter.class, 10L);               
        
        dpdKernel = new Dpd(context);
        globalSizes = new int[]{numberOfDroplets};
        int noOfReductionKernels = 1;
        while (noOfReductionKernels < numberOfDroplets) {
            noOfReductionKernels *= 2;
        }
        localSizes = new int[]{noOfReductionKernels};
    }

    private void performSimulation() {
        
        CLEvent loopEndEvent = initPositionsAndVelocities();
        initDropletParameters();        
//        printVectors("\nPositions", "pos", queue, positions, loopEndEvent);
//        printVectors("\nVelocities", "vel", queue, velocities, loopEndEvent);

        initialRandom = random.nextInt();
        for (step = 1; step <= numberOfSteps; ++step) {
            System.out.println("Step: " + step);
            loopEndEvent = performSingleStep(loopEndEvent);
//            printAverageVelocity(loopEndEvent);
            writePositionsFile(newPositions, loopEndEvent);            
//            printVectors("\nPositions", "pos", queue, newPositions, loopEndEvent);
//            printVectors("\nVelocities", "vel", queue, newVelocities, loopEndEvent);
//            printVectors("\nForces", "force", queue, forces, loopEndEvent);
            swapPositions(loopEndEvent);
            swapVelocities(loopEndEvent);
        }
    }

    private void initDropletParameters(){
        long size = 2L;
        Pointer<DropletParameter> valuesPointer = Pointer.allocateArray(DropletParameter.class, size).order(context.getByteOrder());
        DropletParameter d = new DropletParameter();
        d.temperature(293.1f);
        d.density(3.0f);
        d.repulsionParameter(75.0f);
        d.lambda(0.5f);
        d.sigma(0.075f);
        d.gamma(0.075f * 0.075f / 2.0f / boltzmanConstant / 293.1f);
        valuesPointer.set(0, d);
        valuesPointer.set(1, d);
        dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);        
    }
    
    private CLEvent initPositionsAndVelocities() {        
        CLEvent generatePositionsEvent = dpdKernel.generateTube(queue, positions, types, numberOfDroplets, boxSize,
                    1, random.nextInt(numberOfDroplets), 0.4f, 1.0f, globalSizes, null);
//                dpdKernel.generateRandomVector(queue, positions, boxSize,
//                numberOfDroplets, random.nextInt(numberOfDroplets), globalSizes, null);
        return dpdKernel.generateRandomVector(queue, velocities, boxSize, numberOfDroplets,
                random.nextInt(numberOfDroplets), globalSizes, null, generatePositionsEvent);
    }

    private CLEvent performSingleStep(CLEvent previousStepEvent) {
        CLEvent forcesEvent = calculateForces(previousStepEvent);
        CLEvent newPositionsAndPredictedVelocitiesEvent = calculateNewPositionsAndPredictedVelocities(forcesEvent);
        return calculateNewVelocities(newPositionsAndPredictedVelocitiesEvent);
    }

    private CLEvent calculateForces(CLEvent gaussianRandomsEvent) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, dropletParameters, types, cutoffRadius,
                numberOfDroplets, step + initialRandom, globalSizes, null, gaussianRandomsEvent);
    }

    private CLEvent calculateNewPositionsAndPredictedVelocities(CLEvent forcesEvent) {
        return dpdKernel.calculateNewPositionsAndPredictedVelocities(queue, positions, velocities, forces, newPositions,
                predictedVelocities, deltaTime, lambda, numberOfDroplets, boxSize, globalSizes, null, forcesEvent);
    }

    private CLEvent calculateNewVelocities(CLEvent newPositionsAndPredictedVelocitiesEvent) {
        return dpdKernel.calculateNewVelocities(queue, newPositions, velocities, predictedVelocities, newVelocities,
                forces, dropletParameters, types, deltaTime, cutoffRadius, numberOfDroplets, step + initialRandom, 
                globalSizes, null, newPositionsAndPredictedVelocitiesEvent);
    }

    private void printAverageVelocity(CLEvent... events) {
        CLEvent reductionEvent = dpdKernel.reductionVector(queue, newVelocities, partialSums,
                averageVelocity, numberOfDroplets, localSizes, null, events);
        Pointer<Float> out = averageVelocity.read(queue, reductionEvent);
        System.out.println("avgVel = (" + out.get(0) + ", " + out.get(1) + ", " + out.get(2) + ")");
        out.release();
    }

    private void swapPositions(CLEvent... events) {
        CLEvent.waitFor(events);
        CLBuffer<Float> tmp = positions;
        positions = newPositions;
        newPositions = tmp;
    }

    private void swapVelocities(CLEvent... events) {
        CLEvent.waitFor(events);
        CLBuffer<Float> tmp = velocities;
        velocities = newVelocities;
        newVelocities = tmp;
    }

    private void writePositionsFile(CLBuffer<Float> buffer, CLEvent... events) {
        try (FileWriter writer = new FileWriter("../results/result" + step + ".csv")) {
            Pointer<Float> out = buffer.read(queue, events);
            Pointer<Integer> typesOut = types.read(queue, events);
            writer.write("x, y, z, t\n");
            for (int i = 0; i < numberOfDroplets; i++) {
                writer.write(out.get(i * VECTOR_SIZE) + ","
                        + out.get(i * VECTOR_SIZE + 1) + ","
                        + out.get(i * VECTOR_SIZE + 2) + ","
                        + typesOut.get(i) + "\n");
            }
            out.release();
            typesOut.release();
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
}
