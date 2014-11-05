package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.Random;
import org.bridj.Pointer;
import static pl.edu.agh.student.dpdsimulator.Simulation.*;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public class GpuKernelSimulation extends Simulation {
    
    private String directoryName;
    private int step;
    private Random random;
    
    private CLContext context;
    private CLQueue queue;
    private Dpd dpdKernel;
    private CLBuffer<Integer> cells;
    private CLBuffer<Integer> cellNeighbours;
    private CLBuffer<Float> positions;
    private CLBuffer<Float> newPositions;
    private CLBuffer<Float> velocities;
    private CLBuffer<Float> predictedVelocities;
    private CLBuffer<Float> newVelocities;
    private CLBuffer<Float> forces;
    private CLBuffer<Float> partialSums;
    private CLBuffer<Float> averageVelocity;
    private CLBuffer<Integer> types;
    private CLBuffer<Integer> states;
    private CLBuffer<Dpd.DropletParameter> dropletParameters;
    private Pointer<Float> partialSumsPointer;
    private Pointer<Float> averageVelocityPointer;
    
    @Override
    public void initData() throws IOException {
        context = JavaCL.createBestContext();
        queue = context.createDefaultQueue();

        random = new Random();
        cells = context.createIntBuffer(CLMem.Usage.InputOutput, maxDropletsPerCell * numberOfCells);
        cellNeighbours = context.createIntBuffer(CLMem.Usage.InputOutput,
                numberOfCells * numberOfCellNeighbours);
        positions = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        newPositions = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        velocities = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        predictedVelocities = context.createFloatBuffer(CLMem.Usage.Input,
                numberOfDroplets * VECTOR_SIZE);
        newVelocities = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        forces = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        partialSums = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        partialSumsPointer = Pointer.allocateArray(Float.class,
                numberOfDroplets * VECTOR_SIZE).order(context.getByteOrder());
        averageVelocity = context.createFloatBuffer(CLMem.Usage.InputOutput, VECTOR_SIZE);
        averageVelocityPointer = Pointer.allocateArray(Float.class,
                VECTOR_SIZE).order(context.getByteOrder());
        types = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        states = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);

        dpdKernel = new Dpd(context);

        directoryName = "../results_" + new Date().getTime();
        File directory = new File(directoryName);
        if (!directory.exists()) {
            directory.mkdir();
        }
    }
    
    @Override
    public void performSimulation() {
        long startTime = System.nanoTime();
        step = 0;
        CLEvent loopEndEvent = initSimulationData();
        writePositionsFile(positions, loopEndEvent);
        long endInitTime = System.nanoTime();
        for (step = 1; step <= numberOfSteps; ++step) {
            System.out.println("\nStep: " + step);
            loopEndEvent = performSingleStep(loopEndEvent);
            printAverageVelocity(loopEndEvent);
            writePositionsFile(newPositions, loopEndEvent);
            swapPositions(loopEndEvent);
            swapVelocities(loopEndEvent);
        }
        long endTime = System.nanoTime();
        System.out.println("Init time: " + (endInitTime - startTime) / NANOS_IN_SECOND);
        System.out.println("Mean step time: " + (endTime - startTime) / NANOS_IN_SECOND / numberOfSteps);
    }
    
    private CLEvent initSimulationData() {
        initDropletParameters();
        initStates();
        CLEvent initPositionsAndVelocities = initPositionsAndVelocities();
        CLEvent fillCells = fillCells(positions, initPositionsAndVelocities);
        return fillCellNeighbours(fillCells);
    }
    
    private void initDropletParameters() {
        List<Dpd.DropletParameter> parameters = super.createDropletParameters();
        long size = parameters.size();
        Pointer<Dpd.DropletParameter> valuesPointer
                = Pointer.allocateArray(Dpd.DropletParameter.class, size).order(context.getByteOrder());
        for (int i = 0; i < size; i++) {
            valuesPointer.set(i, parameters.get(i));
        }

        dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }

    private void initStates() {
        Pointer<Integer> statesPointer
                = Pointer.allocateArray(Integer.class, numberOfDroplets).order(context.getByteOrder());
        for (int i = 0; i < numberOfDroplets; i++) {
            statesPointer.set(i, random.nextInt(Integer.MAX_VALUE));
        }

        states = context.createBuffer(CLMem.Usage.InputOutput, statesPointer);
    }

    private CLEvent initPositionsAndVelocities() {
        CLEvent generatePositionsEvent = dpdKernel.generateTube(queue, positions, types,
                states, numberOfDroplets, radiusIn, radiusOut, boxSize, new int[]{numberOfDroplets}, null);
        return dpdKernel.generateRandomVector(queue, velocities, states, types, thermalVelocity,
                flowVelocity, numberOfDroplets, new int[]{numberOfDroplets}, null, generatePositionsEvent);
    }

    private CLEvent fillCells(CLBuffer<Float> positions, CLEvent... events) {
        return dpdKernel.fillCells(queue, cells, positions, cellRadius, boxSize, numberOfDroplets,
                maxDropletsPerCell, numberOfCells, new int[]{numberOfCells}, null, events);
    }

    private CLEvent fillCellNeighbours(CLEvent... events) {
        return dpdKernel.fillCellNeighbours(queue, cellNeighbours, cellRadius, boxSize,
                numberOfCells, new int[]{numberOfCells}, null, events);
    }

    private CLEvent performSingleStep(CLEvent... events) {
        CLEvent forcesEvent = calculateForces(events);
        CLEvent newPositionsAndPredictedVelocitiesEvent
                = calculateNewPositionsAndPredictedVelocities(forcesEvent);
        CLEvent cellsEvent = fillCells(newPositions, newPositionsAndPredictedVelocitiesEvent);
        return calculateNewVelocities(cellsEvent);
    }

    private CLEvent calculateForces(CLEvent... events) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, dropletParameters, types,
                cells, cellNeighbours, cellRadius, boxSize, numberOfDroplets, maxDropletsPerCell, numberOfCells,
                step, new int[]{numberOfDroplets * numberOfCellNeighbours},
                new int[]{numberOfCellNeighbours}, events);
    }

    private CLEvent calculateNewPositionsAndPredictedVelocities(CLEvent... events) {
        return dpdKernel.calculateNewPositionsAndPredictedVelocities(queue, positions, velocities, forces,
                newPositions, predictedVelocities, dropletParameters, types, deltaTime, numberOfDroplets,
                boxSize, new int[]{numberOfDroplets}, null, events);
    }

    private CLEvent calculateNewVelocities(CLEvent... events) {
        return dpdKernel.calculateNewVelocities(queue, newPositions, velocities, predictedVelocities,
                newVelocities, forces, dropletParameters, types, cells, cellNeighbours, deltaTime, 
                cellRadius, boxSize, numberOfDroplets, maxDropletsPerCell, numberOfCells, step, 
                new int[]{numberOfDroplets * numberOfCellNeighbours}, 
                new int[]{numberOfCellNeighbours}, events);
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
        File resultFile = new File(directoryName, "result" + step + ".csv");
        try (FileWriter writer = new FileWriter(resultFile)) {
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

    private void printAverageVelocity(CLEvent... events) {
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            averageVelocityPointer.set(i, 0.0f);
        }

        for (int i = 0; i < numberOfDroplets * VECTOR_SIZE; ++i) {
            partialSumsPointer.set(i, 0.0f);
        }

        CLEvent fillBuffers = averageVelocity.write(queue, averageVelocityPointer, true, events);
        fillBuffers = partialSums.write(queue, partialSumsPointer, true, fillBuffers);
        CLEvent reductionEvent = dpdKernel.doVectorReduction(queue, newVelocities, partialSums,
                averageVelocity, numberOfDroplets, new int[]{numberOfDroplets}, null, fillBuffers);
        Pointer<Float> out = averageVelocity.read(queue, reductionEvent);
        System.out.println("avgVel = (" + out.get(0) + ", " + out.get(1) + ", " + out.get(2) + ")");
        out.release();
    }
}
