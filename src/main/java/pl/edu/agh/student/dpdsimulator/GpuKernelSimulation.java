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
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.bridj.Pointer;
import static pl.edu.agh.student.dpdsimulator.Simulation.*;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public class GpuKernelSimulation extends Simulation {
    
    private static final String SEPARATOR = ",";
    
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
    private CLBuffer<Float> averageVelocity;
    private CLBuffer<Float> averageVelocityPartialSums;
    private CLBuffer<Float> kineticEnergy;
    private CLBuffer<Float> kineticEnergyPartialSums;
    private CLBuffer<Integer> types;
    private CLBuffer<Integer> states;
    private CLBuffer<Dpd.DropletParameter> dropletParameters;
    private Pointer<Float> averageVelocityPointer;
    private Pointer<Float> averageVelocityPartialSumsPointer;
    private Pointer<Float> kineticEnergyPointer;
    private Pointer<Float> kineticEnergyPartialSumsPointer;
    
    @Override
    public void initData(float boxSizeScale, float boxWidthScale, int numberOfDropletsParam) throws IOException {
        context = JavaCL.createBestContext();
        queue = context.createDefaultQueue();

        random = new Random();        
        
        float sizeScale = numberOfDropletsParam / (float)baseNumberOfDroplets;
        numberOfDroplets = numberOfDropletsParam;
        boxSize = (float)Math.cbrt(sizeScale * boxSizeScale / boxWidthScale) * initBoxSize;
        boxWidth = boxWidthScale * boxSize / boxSizeScale;
        radiusIn = boxSize * 0.8f;
        System.out.println("" + boxSize + ", " + boxWidth);
        numberOfCells = (int) (Math.ceil(2 * boxSize / cellRadius) * Math.ceil(2 * boxSize / cellRadius) * Math.ceil(2 * boxWidth / cellRadius));
        
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
        averageVelocity = context.createFloatBuffer(CLMem.Usage.InputOutput, VECTOR_SIZE);
        averageVelocityPointer = Pointer.allocateArray(Float.class,
                VECTOR_SIZE).order(context.getByteOrder());
        averageVelocityPartialSums = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        averageVelocityPartialSumsPointer = Pointer.allocateArray(Float.class,
                numberOfDroplets * VECTOR_SIZE).order(context.getByteOrder());
        kineticEnergy = context.createFloatBuffer(CLMem.Usage.InputOutput, 1);
        kineticEnergyPointer = Pointer.allocateFloat().order(context.getByteOrder());
        kineticEnergyPartialSums = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        kineticEnergyPartialSumsPointer = Pointer.allocateArray(Float.class,
                numberOfDroplets).order(context.getByteOrder());
        types = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        states = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);

        dpdKernel = new Dpd(context);

        if(!shouldStoreFiles) {
            return;
        }
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
//            System.out.println("\nStep: " + step);
            loopEndEvent = performSingleStep(loopEndEvent);
            printAverageVelocity(loopEndEvent);
            writePositionsFile(newPositions, loopEndEvent);
            swapPositions(loopEndEvent);
            swapVelocities(loopEndEvent);
        }
        long endTime = System.nanoTime();
        System.out.println("Init time: " + (endInitTime - startTime) / NANOS_IN_SECOND);
        System.out.println("Mean step time: " + (endTime - startTime) / NANOS_IN_SECOND / numberOfSteps);        
        countSpecial(positions, velocities);
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
                states, numberOfDroplets, radiusIn, boxSize, boxWidth, new int[]{numberOfDroplets}, null);
        return dpdKernel.generateRandomVector(queue, velocities, states, types, thermalVelocity,
                flowVelocity, numberOfDroplets, new int[]{numberOfDroplets}, null, generatePositionsEvent);
    }

    private CLEvent fillCells(CLBuffer<Float> positions, CLEvent... events) {
        return dpdKernel.fillCells(queue, cells, positions, cellRadius, boxSize, boxWidth, numberOfDroplets,
                maxDropletsPerCell, numberOfCells, new int[]{numberOfCells}, null, events);
    }

    private CLEvent fillCellNeighbours(CLEvent... events) {
        return dpdKernel.fillCellNeighbours(queue, cellNeighbours, cellRadius, boxSize, boxWidth,
                numberOfCells, new int[]{numberOfCells}, null, events);
    }

    private CLEvent performSingleStep(CLEvent... events) {
        CLEvent forcesEvent = calculateForces(events);
        CLEvent newPositionsAndPredictedVelocitiesEvent
                = calculateNewPositionsAndPredictedVelocities(forcesEvent);
        CLEvent cellsEvent = fillCells(newPositions, newPositionsAndPredictedVelocitiesEvent);
        CLEvent newVelocitiesEvent = calculateNewVelocities(cellsEvent);
        CLEvent.waitFor(newVelocitiesEvent);
        return newVelocitiesEvent;
    }

    private CLEvent calculateForces(CLEvent... events) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, dropletParameters, types,
                cells, cellNeighbours, cellRadius, boxSize, boxWidth, numberOfDroplets, maxDropletsPerCell, 
                numberOfCells, step, new int[]{numberOfDroplets * numberOfCellNeighbours},
                new int[]{numberOfCellNeighbours}, events);
    }

    private CLEvent calculateNewPositionsAndPredictedVelocities(CLEvent... events) {
        return dpdKernel.calculateNewPositionsAndPredictedVelocities(queue, positions, velocities, forces,
                newPositions, predictedVelocities, dropletParameters, types, deltaTime, numberOfDroplets, 
                boxSize, boxWidth, new int[]{numberOfDroplets}, null, events);
    }

    private CLEvent calculateNewVelocities(CLEvent... events) {
        return dpdKernel.calculateNewVelocities(queue, newPositions, velocities, predictedVelocities,
                newVelocities, forces, dropletParameters, types, cells, cellNeighbours, deltaTime, 
                cellRadius, boxSize, boxWidth, numberOfDroplets, maxDropletsPerCell, numberOfCells, step, 
                new int[]{numberOfDroplets * numberOfCellNeighbours}, new int[]{numberOfCellNeighbours}, events);
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
        if(!shouldStoreFiles) {
            return;
        }
        File resultFile = new File(directoryName, "result" + step + ".csv");
        try (FileWriter writer = new FileWriter(resultFile)) {
            Pointer<Float> out = buffer.read(queue, events);
            Pointer<Integer> typesOut = types.read(queue, events);
            writer.write("x, y, z, t\n");
            for (int i = 0; i < numberOfDroplets; i++) {
                writer.write(out.get(i * VECTOR_SIZE) + SEPARATOR
                        + out.get(i * VECTOR_SIZE + 1) + SEPARATOR
                        + out.get(i * VECTOR_SIZE + 2) + SEPARATOR
                        + typesOut.get(i) + "\n");
            }
            out.release();
            typesOut.release();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void countSpecial(CLBuffer<Float> positions, CLBuffer<Float> velocities, CLEvent... events) {
        final double sliceSize = boxSize / 10;
        Pointer<Float> positionsPointer = positions.read(queue, events);
        Set[] buckets = new Set[(int)(2 * boxSize / sliceSize) + 1];
        for(int i = 0; i < buckets.length; ++i) {
            buckets[i] = new HashSet();
        }
        
        for (int i = 0; i < numberOfDroplets; i++) {
            float x = positionsPointer.get(i * VECTOR_SIZE);
            float y = positionsPointer.get(i * VECTOR_SIZE + 1);
            float z = positionsPointer.get(i * VECTOR_SIZE + 2);
            if(Math.abs(x) <= sliceSize && Math.abs(y) <= sliceSize) {
                buckets[(int)((z + boxSize)/sliceSize)].add(i);
            }
        }
        
        Pointer<Float> velocitiesPointer = velocities.read(queue, events);
        float[] meanVels = new float[buckets.length];
        for(int i = 0; i < buckets.length; ++i) {
            float velSum = 0f;
            for(Object v : buckets[i]) {
                int index = (Integer) v;
                velSum += velocitiesPointer.get(index * VECTOR_SIZE + 1);
            }
            if(buckets[i].isEmpty()) {
                meanVels[i] = 0;
            } else {
                meanVels[i] = velSum / buckets[i].size();
            }
        }
        
        for(int i = 0; i < buckets.length; ++i) {
            System.out.println(meanVels[i]);
        }
                
        positionsPointer.release();
        velocitiesPointer.release();
    }

    private void printAverageVelocity(CLEvent... events) {
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            averageVelocityPointer.set(i, 0.0f);
        }
        kineticEnergyPointer.set(0.0f);
        
        for (int i = 0; i < numberOfDroplets * VECTOR_SIZE; ++i) {
            averageVelocityPartialSumsPointer.set(i, 0.0f);
            kineticEnergyPartialSumsPointer.set(i / VECTOR_SIZE, 0.0f);
        }
        
        CLEvent fillBuffers = averageVelocity.write(queue, averageVelocityPointer, true, events);
        fillBuffers = averageVelocityPartialSums.write(queue, averageVelocityPartialSumsPointer, true, fillBuffers);
        fillBuffers = kineticEnergy.write(queue, kineticEnergyPointer, true, fillBuffers);
        fillBuffers = kineticEnergyPartialSums.write(queue, kineticEnergyPartialSumsPointer, true, fillBuffers);
        CLEvent reductionEvent = dpdKernel.calculateAvgVelocityAndEnergy(queue, newVelocities, 
                averageVelocityPartialSums, averageVelocity, kineticEnergyPartialSums, kineticEnergy, 
                numberOfDroplets, new int[]{numberOfDroplets}, null, fillBuffers);
        Pointer<Float> avgVelocityOut = averageVelocity.read(queue, reductionEvent);
        Pointer<Float> kineticEnergyOut = kineticEnergy.read(queue, reductionEvent);
        System.out.println(/*"avgVel = (" + 
                avgVelocityOut.get(0) + ", " + 
                avgVelocityOut.get(1) + ", " + 
                avgVelocityOut.get(2) + "); en_k = " +*/
                kineticEnergyOut.get(0)                
        );
        avgVelocityOut.release();
        kineticEnergyOut.release();
    }    
}
