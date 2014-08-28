package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
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
    private Pointer<Integer> typesPointer;
    private static List<DropletParameter> parameters = new ArrayList<>();

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
        //dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, DropletParameter.class, 10L);               

        dpdKernel = new Dpd(context);
        globalSizes = new int[]{numberOfDroplets};
        int noOfReductionKernels = 1;
        while (noOfReductionKernels < numberOfDroplets) {
            noOfReductionKernels *= 2;
        }
        localSizes = new int[]{noOfReductionKernels};
    }

    private void performSimulation() {
        step = 0;
        initDropletParameters();

        CLEvent loopEndEvent = initPositionsAndVelocities();
        writePositionsFile(positions, loopEndEvent);
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

    private void initDropletParameters() {    
        float lambda = 0.63f;
        float sigma = 0.075f;
        float gamma = sigma * sigma / 2.0f / boltzmanConstant / temperature;
        float velocityInitRange = 0.0f;
        
        //naczynie
        float density = 12.0f;
        float repulsionParameter = 75.0f * boltzmanConstant * temperature / density;    
        addParameter(4, density, repulsionParameter, lambda, sigma, gamma, 0.0f);
        
        density = 3.0f;
        repulsionParameter = 75.0f * boltzmanConstant * temperature / density;   
        //krwinka
        addParameter(1.14f, density, repulsionParameter, lambda, sigma, gamma, velocityInitRange);
        //osocze
        addParameter(1, density, repulsionParameter, lambda, sigma, gamma, velocityInitRange);
        
        long size = parameters.size();
        Pointer<DropletParameter> valuesPointer = Pointer.allocateArray(DropletParameter.class, size).order(context.getByteOrder());
        for (int i = 0; i < size; i++) {
            valuesPointer.set(i, parameters.get(i));
        }
        
        dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }

    private CLEvent initPositionsAndVelocities() {
//        CLEvent generatePositionsEvent = dpdKernel.generateTube(queue, positions, types, numberOfDroplets, 
//                random.nextInt(numberOfDroplets), 0.4f * boxSize, 0.5f * boxSize, boxSize, globalSizes, null);
//        dpdKernel.generateVelocities(queue, velocities, dropletParameters, types, numberOfDroplets,
//                random.nextInt(numberOfDroplets), globalSizes, null);//, generatePositionsEvent
        long t = System.nanoTime();
        generateTube();
        generateVelocities();
        System.out.println("time: " + (System.nanoTime() - t));
        return null;
    }
    
    private void generateTube() {
        int numberOfCoordinates = numberOfDroplets * VECTOR_SIZE;
        typesPointer = Pointer.allocateArray(Integer.class, numberOfDroplets);
        Pointer<Float> positionsPointer = Pointer.allocateArray(Float.class, numberOfCoordinates);
        float radiusIn = 0.8f * boxSize;
        float radiusOut = 0.95f * boxSize;
        for (int i = 0; i < numberOfCoordinates; i += VECTOR_SIZE) {
            float x = nextRandomFloat(radiusOut);
            float z = nextRandomFloat((float) Math.sqrt(radiusOut * radiusOut - x * x));
            float y = nextRandomFloat(boxSize);
            
            positionsPointer.set(i, x);
            positionsPointer.set(i + 1, y);
            positionsPointer.set(i + 2, z);
            positionsPointer.set(i + 3, 0.0f);
            
            float distanceFromY = (float) Math.sqrt(x * x + z * z);
            int dropletId = i / 4;
            if (distanceFromY >= radiusIn) {
                typesPointer.set(dropletId, 0);
            } else {
                float randomNum = nextRandomFloat(1);
                if (randomNum >= 0.0f) {
                    typesPointer.set(dropletId, 1);
                } else {
                    typesPointer.set(dropletId, 2);
                }        
            }
        }
        positions = context.createBuffer(CLMem.Usage.InputOutput, positionsPointer);
        types = context.createBuffer(CLMem.Usage.InputOutput, typesPointer);
    }
    
    private void generateVelocities() {
        int numberOfCoordinates = numberOfDroplets * VECTOR_SIZE;
        Pointer<Float> velocitiesPointer = Pointer.allocateArray(Float.class, numberOfCoordinates);
        for (int i = 0; i < numberOfCoordinates; i += VECTOR_SIZE) {
            int dropletId = i / 4;
            float velocityInitRange = parameters.get(typesPointer.get(dropletId)).velocityInitRange();
            float x = nextRandomFloat(velocityInitRange);
            float y = nextRandomFloat(velocityInitRange);
            float z = nextRandomFloat(velocityInitRange);            
            
            velocitiesPointer.set(i, x);
            velocitiesPointer.set(i + 1, y);
            velocitiesPointer.set(i + 2, z);
            velocitiesPointer.set(i + 3, 0.0f);
        }
        velocities = context.createBuffer(CLMem.Usage.InputOutput, velocitiesPointer);
    }
        
    public void addParameter(float mass, float density, float repulsionParameter, float lambda, float sigma, float gamma, float velocityInitRange) {
        DropletParameter dropletParameter = new DropletParameter();
        dropletParameter.mass(mass);
        dropletParameter.density(density);
        dropletParameter.repulsionParameter(repulsionParameter);
        dropletParameter.lambda(lambda);
        dropletParameter.sigma(sigma);
        dropletParameter.gamma(gamma);
        dropletParameter.velocityInitRange(velocityInitRange);

        parameters.add(dropletParameter);
    }

    private float nextRandomFloat(float range) {
        return random.nextFloat() * 2 * range - range;
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
                predictedVelocities, dropletParameters, types, deltaTime, numberOfDroplets, boxSize, globalSizes, null, forcesEvent);
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

    private void printParams() {
        Pointer<DropletParameter> out = dropletParameters.read(queue);
        for (int i = 0; i < 2; i++) {
            System.out.println(i + ": " + out.get(i).gamma());
        }
    }
}
