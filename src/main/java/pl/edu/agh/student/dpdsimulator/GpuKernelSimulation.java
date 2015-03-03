package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.util.OpenCLType;
import com.nativelibs4java.opencl.util.ReductionUtils;
import com.nativelibs4java.opencl.util.ReductionUtils.Reductor;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import org.bridj.Pointer;
import static pl.edu.agh.student.dpdsimulator.Simulation.*;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public class GpuKernelSimulation extends Simulation {
    
    private static final String SEPARATOR = ",";
    
    private String directoryName;
    private String dataFileName = "simulation.data";
    
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
    private CLBuffer<Float> newVelocities;
    private CLBuffer<Float> forces;
    private CLBuffer<Integer> states;
    private CLBuffer<Integer> types;
    private CLBuffer<Dpd.Parameters> parameters;
    private Reductor<Float> sumator;
    private Reductor<Float> sumatorVector;
    private CLBuffer<Float> velocitiesEnergy;
    
    @Override
    public void initData() throws IOException {
        context = JavaCL.createContext(null, JavaCL.listGPUPoweredPlatforms()[0].getBestDevice());
        queue = context.createDefaultQueue();

        random = new Random();        
        
        
        float sizeScale = numberOfDroplets / (float)baseNumberOfDroplets;        
        boxSize = (float)Math.cbrt(sizeScale * boxSizeScale / boxWidthScale) * initBoxSize;
        boxWidth = boxWidthScale * boxSize / boxSizeScale;
        radiusIn = boxSize * 0.8f;
        numberOfCells = (int) (Math.ceil(2 * boxSize / cellRadius) * Math.ceil(2 * boxSize / cellRadius) * Math.ceil(2 * boxWidth / cellRadius));
        
        System.out.println("" + boxSize + ", " + boxWidth + "; " + numberOfDroplets + "; " + numberOfCells);
        
        cells = context.createIntBuffer(CLMem.Usage.InputOutput, maxDropletsPerCell * numberOfCells);
        cellNeighbours = context.createIntBuffer(CLMem.Usage.InputOutput,
                numberOfCells * numberOfCellNeighbours);
        positions = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        newPositions = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        velocities = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        newVelocities = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        forces = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        states = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);     
        types = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);   
        velocitiesEnergy = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        sumator = ReductionUtils.createReductor(context, ReductionUtils.Operation.Add, OpenCLType.Float, 1);
        sumatorVector = ReductionUtils.createReductor(context, ReductionUtils.Operation.Add, OpenCLType.Float, VECTOR_SIZE);
        
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
        List<Dpd.Parameters> params = super.loadParametersFromFile(dataFileName);
        long size = params.size();
        Pointer<Dpd.Parameters> valuesPointer
                = Pointer.allocateArray(Dpd.Parameters.class, size).order(context.getByteOrder());
        for (int i = 0; i < size; i++) {
            valuesPointer.set(i, params.get(i));
        }

        parameters = context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }

    private void initStates() {
        Pointer<Integer> statesPointer
                = Pointer.allocateArray(Integer.class, numberOfDroplets).order(context.getByteOrder());
        for (int i = 0; i < numberOfDroplets; i++) {
            statesPointer.set(i, random.nextInt());
        }

        states = context.createBuffer(CLMem.Usage.InputOutput, statesPointer);
    }

    private CLEvent initPositionsAndVelocities() {
        CLEvent generatePositionsEvent = dpdKernel.generateTube(queue, positions, types,
                states, numberOfDroplets, radiusIn, boxSize, boxWidth, new int[]{numberOfDroplets}, null);
        return dpdKernel.generateRandomVector(queue, velocities, states, types, numberOfDroplets, 
                new int[]{numberOfDroplets}, null, generatePositionsEvent);
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
        CLEvent newPositionsAndVelocitiesEvent = calculateNewPositionsAndVelocities(forcesEvent);
        CLEvent cellsEvent = fillCells(newPositions, newPositionsAndVelocitiesEvent);
        return cellsEvent;
    }

    private CLEvent calculateForces(CLEvent... events) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, parameters, types,
                cells, cellNeighbours, cellRadius, boxSize, boxWidth, numberOfDroplets, maxDropletsPerCell, 
                numberOfCells, step, new int[]{numberOfDroplets}, null, events);
    }

    private CLEvent calculateNewPositionsAndVelocities(CLEvent... events) {
        return dpdKernel.calculateNewPositionsAndVelocities(queue, positions, velocities, forces,
                newPositions, newVelocities, parameters, types, deltaTime, numberOfDroplets, 
                boxSize, boxWidth, new int[]{numberOfDroplets}, null, events);
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

    private void countSpecial(CLBuffer<Float> positions, CLBuffer<Float> velocities, CLEvent... events) {
        final double sliceSize = cellRadius;
        Pointer<Float> positionsPointer = positions.read(queue, events);
        Set[] buckets = new Set[(int)Math.ceil(2 * boxSize / sliceSize)];
        for(int i = 0; i < buckets.length; ++i) {
            buckets[i] = new HashSet();
        }
        
        for (int i = 0; i < numberOfDroplets; i++) {
            float x = positionsPointer.get(i * VECTOR_SIZE);
            float y = positionsPointer.get(i * VECTOR_SIZE + 1);
            float z = positionsPointer.get(i * VECTOR_SIZE + 2);
            //if(Math.abs(x) <= sliceSize && Math.abs(y) <= sliceSize) {
                buckets[(int)((z + boxSize)/sliceSize)].add(i);
            //}
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
            System.out.println(String.format(Locale.GERMANY, "%e", meanVels[i]));
        }
                
        positionsPointer.release();
        velocitiesPointer.release();
    }

    private void printAverageVelocity(CLEvent... events) {        
//        Pointer<Float> velocitiesSum = sumatorVector.reduce(queue, newVelocities, events);
//        float avgVelocityX = velocitiesSum.get(0) / numberOfDroplets;
//        float avgVelocityY = velocitiesSum.get(1) / numberOfDroplets;
//        float avgVelocityZ = velocitiesSum.get(2) / numberOfDroplets;
        CLEvent reductionEvent = dpdKernel.calculateVelocitiesEnergy(queue, newVelocities, velocitiesEnergy, 
                numberOfDroplets, new int[]{numberOfDroplets}, null, events);
        Pointer<Float> kineticEnergySum = sumator.reduce(queue, velocitiesEnergy, reductionEvent);
        float ek = kineticEnergySum.get(0) / numberOfDroplets;
        System.out.println(
                /*"avgVel = (" + 
                avgVelocityX + ", " + 
                avgVelocityY + ", " + 
                avgVelocityZ + "); en_k = " +*/
                String.format(Locale.GERMANY, "%e", ek)
        );
//        velocitiesSum.release();
        kineticEnergySum.release();
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
}
