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
import java.util.Date;
import java.util.HashSet;
import java.util.Locale;
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
    private CLBuffer<Float> newVelocities;
    private CLBuffer<Float> forces;
    private CLBuffer<Integer> states;
    private CLBuffer<Integer> types;
    private CLBuffer<Dpd.SimulationParameters> simulationParameters;
    private CLBuffer<Dpd.DropletParameters> dropletParameters;
    private CLBuffer<Dpd.PairParameters> pairParameters;    
    private Reductor<Float> sumator;
    private CLBuffer<Float> velocitiesEnergy;
    
    @Override
    public void initData() throws Exception {
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
        
        dpdKernel = new Dpd(context);

        if(!shouldStoreFiles) {
            return;
        }
        directoryName = resultsDirectoryBase + new Date().getTime();
        File directory = new File(directoryName);
        if (!directory.exists()) {
            directory.mkdir();
        }
    }
    
    @Override
    public void performSimulation() throws Exception {                              
        long startTime = System.nanoTime();
        step = 0;
        CLEvent loopEndEvent = initSimulationData();
        writeDataFile(positions, velocities, loopEndEvent);   
        printAverageVelocity(velocities, loopEndEvent); 
        printKineticEnergy(velocities, loopEndEvent);
        long endInitTime = System.nanoTime();
        for (step = 1; step <= numberOfSteps; ++step) {
            System.out.println("\nStep: " + step);
            loopEndEvent = performSingleStep(loopEndEvent);
            printAverageVelocity(newVelocities, loopEndEvent);
            printKineticEnergy(newVelocities, loopEndEvent);
            writeDataFile(newPositions, newVelocities, loopEndEvent);
            swapPositions(loopEndEvent);
            swapVelocities(loopEndEvent);
        }
        long endTime = System.nanoTime();
        System.out.println("Init time: " + (endInitTime - startTime) / NANOS_IN_SECOND);
        System.out.println("Mean step time: " + (endTime - startTime) / NANOS_IN_SECOND / numberOfSteps);        
        printVelocityProfile(positions, velocities);
    }
    
    private CLEvent initSimulationData() throws Exception {
        initParameters();
        initStates();
        CLEvent initPositionsAndVelocities = initPositionsAndVelocities();
        CLEvent fillCells = fillCells(positions, initPositionsAndVelocities);
        return fillCellNeighbours(fillCells);
    }
    
    private void initParameters() throws Exception {
        super.loadParametersFromFile(dataFileName);
        Pointer<Dpd.SimulationParameters> simulationParametersPointer = Pointer.allocate(Dpd.SimulationParameters.class);
        simulationParametersPointer.set(createSimulationParameter());
        simulationParameters = context.createBuffer(CLMem.Usage.InputOutput, simulationParametersPointer);
        pairParameters = context.createBuffer(CLMem.Usage.InputOutput, pairParametersPointer);
        dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, dropletParametersPointer);
    }
    
    private Dpd.SimulationParameters createSimulationParameter() {
        return new Dpd.SimulationParameters()
                .boxSize(boxSize)
                .boxWidth(boxWidth)
                .numberOfDroplets(numberOfDroplets)
                .numberOfCells(numberOfCells)
                .numberOfTypes(numberOfCellKinds)
                .maxDropletsPerCell(maxDropletsPerCell)
                .deltaTime(deltaTime)
                .cellRadius(cellRadius)
                .radiusIn(radiusIn);
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
        CLEvent generatePositionsEvent;
        if(generateRandomPositions){
            generatePositionsEvent = dpdKernel.generateRandomPositions(queue, positions, types,
                states, simulationParameters, new int[]{numberOfDroplets}, null);
        } else {
            generatePositionsEvent = dpdKernel.generateTube(queue, positions, types,
                states, simulationParameters, new int[]{numberOfDroplets}, null);        
        }
        return dpdKernel.generateVelocities(queue, velocities, states, types, simulationParameters, 
                new int[]{numberOfDroplets}, null, generatePositionsEvent);
    }

    private CLEvent fillCells(CLBuffer<Float> positions, CLEvent... events) {
        return dpdKernel.fillCells(queue, cells, positions, simulationParameters, new int[]{numberOfCells}, null, events);
    }

    private CLEvent fillCellNeighbours(CLEvent... events) {
        return dpdKernel.fillCellNeighbours(queue, cellNeighbours, simulationParameters, new int[]{numberOfCells}, null, events);
    }

    private CLEvent performSingleStep(CLEvent... events) {
        CLEvent forcesEvent = calculateForces(events);
        CLEvent newPositionsAndVelocitiesEvent = calculateNewPositionsAndVelocities(forcesEvent);
        CLEvent cellsEvent = fillCells(newPositions, newPositionsAndVelocitiesEvent);
        return cellsEvent;
    }

    private CLEvent calculateForces(CLEvent... events) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, types,
                cells, cellNeighbours, pairParameters, dropletParameters, simulationParameters, 
                step, new int[]{numberOfDroplets}, null, events);
    }

    private CLEvent calculateNewPositionsAndVelocities(CLEvent... events) {
        return dpdKernel.calculateNewPositionsAndVelocities(queue, positions, velocities, forces,
                newPositions, newVelocities, types, pairParameters, dropletParameters, 
                simulationParameters, new int[]{numberOfDroplets}, null, events);
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

    private void printAverageVelocity(CLBuffer<Float> velocities, CLEvent... events) {      
        if(!shouldPrintAvgVelocity) {
            return;
        }
        Pointer<Float> velocitiesOut = velocities.read(queue, events);
        float velocitiesSumX = 0f, velocitiesSumY = 0f, velocitiesSumZ = 0f;
        for(int i = 0; i < numberOfDroplets; ++i) {
            velocitiesSumX += velocitiesOut.get(i * VECTOR_SIZE);
            velocitiesSumY += velocitiesOut.get(i * VECTOR_SIZE + 1);
            velocitiesSumZ += velocitiesOut.get(i * VECTOR_SIZE + 2);
        }
        float avgVelocityX = velocitiesSumX / numberOfDroplets;
        float avgVelocityY = velocitiesSumY / numberOfDroplets;
        float avgVelocityZ = velocitiesSumZ / numberOfDroplets;
        System.out.println("avgVelocity=" 
                + avgVelocityX + SEPARATOR
                + avgVelocityY + SEPARATOR 
                + avgVelocityZ);
        velocitiesOut.release();
    }

    private void printKineticEnergy(CLBuffer<Float> velocities, CLEvent... events) {
        if(!shouldPrintKineticEnergy) {
            return;
        }        
        CLEvent reductionEvent = dpdKernel.calculateVelocitiesEnergy(queue, velocities, velocitiesEnergy, 
                simulationParameters, new int[]{numberOfDroplets}, null, events);
        Pointer<Float> kineticEnergySum = sumator.reduce(queue, velocitiesEnergy, reductionEvent);
        float ek = kineticEnergySum.get(0) / numberOfDroplets;
        System.out.println("Ek=" + String.format(Locale.GERMANY, "%e", ek));
        kineticEnergySum.release();
    }

    private void writeDataFile(CLBuffer<Float> positions, CLBuffer<Float> velocities, CLEvent... events) {
        if(!shouldStoreFiles || step % stepDumpThreshold != 0) {
            return;
        }
        File resultFile = new File(directoryName, "result" + step + ".csv");
        try (FileWriter writer = new FileWriter(resultFile)) {
            Pointer<Float> positionsOut = positions.read(queue, events);
            Pointer<Float> velocitiesOut = velocities.read(queue, events);
            Pointer<Integer> typesOut = types.read(queue, events);
            writer.write("x,y,z,vx,vy,vz,t\n");
            for (int i = 0; i < numberOfDroplets; i++) {
                writer.write(
                        positionsOut.get(i * VECTOR_SIZE) + SEPARATOR
                        + positionsOut.get(i * VECTOR_SIZE + 1) + SEPARATOR
                        + positionsOut.get(i * VECTOR_SIZE + 2) + SEPARATOR
                        + velocitiesOut.get(i * VECTOR_SIZE) + SEPARATOR
                        + velocitiesOut.get(i * VECTOR_SIZE + 1) + SEPARATOR
                        + velocitiesOut.get(i * VECTOR_SIZE + 2) + SEPARATOR
                        + typesOut.get(i) + "\n");
            }
            positionsOut.release();
            velocitiesOut.release();
            typesOut.release();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void printVelocityProfile(CLBuffer<Float> positions, CLBuffer<Float> velocities, CLEvent... events) {
        if(!shouldPrintVelocityProfile) {
            return;
        }
        
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
}
