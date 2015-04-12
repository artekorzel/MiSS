package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.LocalSize;
import java.io.File;
import java.io.FileWriter;
import static java.nio.file.Files.copy;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import java.util.Date;
import java.util.HashSet;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import org.bridj.Pointer;
import static pl.edu.agh.student.dpdsimulator.Simulation.*;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public class GpuKernelSimulation extends Simulation {
    
    private static final String CSVSEPARATOR = ",";
    private static final String PSISEPARATOR = " ";
    
    private String directoryName;
    private int[] numberOfDropletsPerType;
    
    private int step;
    private Random random;
    
    private CLContext context;
    private CLQueue queue;
    private Dpd dpdKernel;
    private CLBuffer<Integer> cells;
    private CLBuffer<Integer> cellNeighbours;
    private CLBuffer<Float> positions;
    private CLBuffer<Float> velocities;
    private CLBuffer<Float> forces;
    private CLBuffer<Integer> states;
    private CLBuffer<Integer> types;
    private CLBuffer<Integer> dropletsPerType;
    private CLBuffer<Dpd.SimulationParameters> simulationParameters;
    private CLBuffer<Dpd.DropletParameters> dropletParameters;
    private CLBuffer<Dpd.PairParameters> pairParameters;
    private CLBuffer<Float> partialEnergy;
    private CLBuffer<Float> partialAverageVelocity;
    
    @Override
    public void initData() throws Exception {
        context = JavaCL.createContext(null, JavaCL.listGPUPoweredPlatforms()[0].getBestDevice());
        queue = context.createDefaultQueue();
        random = new Random();
        
        cells = context.createIntBuffer(CLMem.Usage.InputOutput, maxDropletsPerCell * numberOfCells);
        cellNeighbours = context.createIntBuffer(CLMem.Usage.InputOutput,
                numberOfCells * numberOfCellNeighbours);
        positions = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        velocities = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        forces = context.createFloatBuffer(CLMem.Usage.InputOutput,
                numberOfDroplets * VECTOR_SIZE);
        states = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        types = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        dropletsPerType = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfCellKinds);
        partialEnergy = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfReductionGroups);
        partialAverageVelocity = context.createFloatBuffer(CLMem.Usage.InputOutput, 
                numberOfReductionGroups * VECTOR_SIZE);
        
        dpdKernel = new Dpd(context);

        if(!(shouldStoreCSVFiles || shouldStorePSIFiles)) {
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
        writeDataFile(loopEndEvent);   
        printAverageVelocity(loopEndEvent); 
        printKineticEnergy(loopEndEvent);
        long endInitTime = System.nanoTime();
        for (step = 1; step <= numberOfSteps; ++step) {
            System.out.println("\nStep: " + step);
            loopEndEvent = performSingleStep(loopEndEvent);
            printAverageVelocity(loopEndEvent);
            printKineticEnergy(loopEndEvent);
            writeDataFile(loopEndEvent);
            CLEvent.waitFor(loopEndEvent);
        }
        long endTime = System.nanoTime();
        System.out.println("\nInit time: " + (endInitTime - startTime) / NANOS_IN_SECOND);
        System.out.println("Mean step time: " + (endTime - startTime) / NANOS_IN_SECOND / numberOfSteps);   
        copy(new File("simulation.data").toPath(), new File(directoryName + "/simulation.data").toPath(), REPLACE_EXISTING);
        printVelocityProfile();
    }
    
    private CLEvent initSimulationData() throws Exception {
        initParameters();
        initStates();
        CLEvent initPositionsAndVelocities = initPositionsAndVelocities();
        CLEvent fillCells = fillCells(positions, initPositionsAndVelocities);
        return fillCellNeighbours(fillCells);
    }
    
    private void initParameters() throws Exception {
        Pointer<Dpd.SimulationParameters> simulationParametersPointer = 
                Pointer.allocate(Dpd.SimulationParameters.class).order(context.getByteOrder());        
        Pointer<Dpd.DropletParameters> dropletParametersPointer =
                Pointer.allocateArray(Dpd.DropletParameters.class, numberOfCellKinds);
        Pointer<Dpd.PairParameters> pairParametersPointer =
                Pointer.allocateArray(Dpd.PairParameters.class, numberOfCellKinds * numberOfCellKinds);
        
        simulationParametersPointer.set(createSimulationParameter());
        dropletParametersPointer.setArray(createDropletParameters());
        pairParametersPointer.setArray(createPairParameters());
        
        simulationParameters = context.createBuffer(CLMem.Usage.InputOutput, simulationParametersPointer);
        pairParameters = context.createBuffer(CLMem.Usage.InputOutput, pairParametersPointer);
        dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, dropletParametersPointer);
        
        simulationParametersPointer.release();
        pairParametersPointer.release();
        dropletParametersPointer.release();
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
                .radiusIn(radiusIn)
                .accelerationVesselPart(2 * accelerationVesselPart * boxWidth - boxWidth)
                .accelerationValue(accelerationValue)
                .averageDropletDistance(averageDropletDistance)
                .accelerationVeselSteps(accelerationVeselSteps)
                .shouldSimulateVesselDroplets(shouldSimulateVesselDroplets);
    }

    private Dpd.DropletParameters[] createDropletParameters() {                
        Dpd.DropletParameters[] dropletParametersArray = new Dpd.DropletParameters[numberOfCellKinds];
        for(int i = 0; i < numberOfCellKinds; i++){                                
            dropletParametersArray[i] = createDropletParameter(mass[i]);
        }
        return dropletParametersArray;
    }

    private Dpd.PairParameters[] createPairParameters() {        
        Dpd.PairParameters[] pairParametersArray = new Dpd.PairParameters[numberOfCellKinds * numberOfCellKinds];
        for(int i = 0, counter = 0; i < numberOfCellKinds; i++){
            for(int j = 0; j < numberOfCellKinds; j++, counter++){
                pairParametersArray[counter] = createPairParameter(cutOffRadius[i][j], pi[i][j], gamma[i][j], sigma[i][j]);
            }
        }
        return pairParametersArray;
    }

    private void initStates() {
        Pointer<Integer> statesPointer
                = Pointer.allocateArray(Integer.class, numberOfDroplets).order(context.getByteOrder());
        for (int i = 0; i < numberOfDroplets; i++) {
            statesPointer.set(i, random.nextInt());
        }

        states = context.createBuffer(CLMem.Usage.InputOutput, statesPointer);
        statesPointer.release();
    }

    private CLEvent initPositionsAndVelocities() {
        CLEvent generatePositionsEvent;
        if(generateRandomPositions){
            generatePositionsEvent = dpdKernel.generateBoryczko(queue, positions, types, states,
                    cellsXAxis, cellsYAxis, cellsZAxis, simulationParameters, new int[]{1}, null);
//            generatePositionsEvent = dpdKernel.generateRandomPositions(queue, positions, types,
//                states, simulationParameters, new int[]{numberOfDroplets}, null);
        } else {
            generatePositionsEvent = dpdKernel.generateTube(queue, positions, types,
                states, simulationParameters, new int[]{numberOfDroplets}, null);        
        }
        CLEvent countDropletsEvent = dpdKernel.countDropletsPerType(queue, types, dropletsPerType, 
                simulationParameters, new int[]{numberOfCellKinds}, null, generatePositionsEvent);
        Pointer<Integer> dropletsPerTypePointer = dropletsPerType.read(queue, countDropletsEvent);
        numberOfDropletsPerType = dropletsPerTypePointer.getInts();
        dropletsPerTypePointer.release();
        return dpdKernel.generateVelocities(queue, velocities, states, types, simulationParameters, 
                new int[]{numberOfDroplets}, null, countDropletsEvent);
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
        CLEvent cellsEvent = fillCells(positions, newPositionsAndVelocitiesEvent);
        return cellsEvent;
    }

    private CLEvent calculateForces(CLEvent... events) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, types,
                cells, cellNeighbours, pairParameters, dropletParameters, simulationParameters, 
                step, new int[]{numberOfDroplets}, null, events);
    }

    private CLEvent calculateNewPositionsAndVelocities(CLEvent... events) {
        return dpdKernel.calculateNewPositionsAndVelocities(queue, positions, velocities, forces, types, pairParameters,
                dropletParameters, simulationParameters, new int[]{numberOfDroplets}, null, events);
    }

    private void printAverageVelocity(CLEvent... events) {      
        if(!shouldPrintAvgVelocity) {
            return;
        }
        CLEvent reductionEvent = dpdKernel.calculateAverageVelocity(queue, velocities,
                LocalSize.ofFloatArray(reductionLocalSize * VECTOR_SIZE), partialAverageVelocity, 
                simulationParameters, new int[]{reductionSize}, new int[]{reductionLocalSize}, events);
        Pointer<Float> avgVelocity = partialAverageVelocity.read(queue, reductionEvent);
        float avgVelocityX = 0, avgVelocityY = 0, avgVelocityZ = 0;
        for(int i = 0; i < numberOfReductionGroups; ++i) {
            avgVelocityX += avgVelocity.get(i * VECTOR_SIZE);
            avgVelocityY += avgVelocity.get(i * VECTOR_SIZE + 1);
            avgVelocityZ += avgVelocity.get(i * VECTOR_SIZE + 2);
        }        
        avgVelocityX /= numberOfDroplets;
        avgVelocityY /= numberOfDroplets;
        avgVelocityZ /= numberOfDroplets;
        System.out.println("avgVelocity=" 
                + avgVelocityX + PSISEPARATOR
                + avgVelocityY + PSISEPARATOR 
                + avgVelocityZ);
        avgVelocity.release();
    }

    private void printKineticEnergy(CLEvent... events) {
        if(!shouldPrintKineticEnergy) {
            return;
        }
        System.out.print("Ek=");
        float[] ek = new float[numberOfCellKinds];
        Pointer<Float> partialEnergyPointer = Pointer.allocateFloats(numberOfReductionGroups).order(context.getByteOrder());
        for(int type = 0; type < numberOfCellKinds; ++type) {
            CLEvent reductionEvent = dpdKernel.calculateKineticEnergy(queue, velocities,
                    LocalSize.ofFloatArray(reductionLocalSize), partialEnergy, types, dropletParameters, 
                    simulationParameters, type, new int[]{reductionSize}, new int[]{reductionLocalSize}, events);
            partialEnergy.read(queue, partialEnergyPointer, true, reductionEvent);
            ek[type] = 0;
            for(float energy : partialEnergyPointer.getFloats()) {
                ek[type] += energy;
            }
            System.out.print(ek[type] + " ");
        }
        System.out.println();
        partialEnergyPointer.release();
        
        System.out.print("T=");
        for(int type = 0; type < numberOfCellKinds; ++type) {
            float fee = (float) (fe * numberOfDroplets / numberOfDropletsPerType[type]);
            float temp = (float) (ft * (fee * ek[type]));
            System.out.print(temp + " ");
        }
        System.out.println();
    }

    private void writeDataFile(CLEvent... events) {
        if(shouldStoreCSVFiles){
            writeDataFile(csvHeader, CSVSEPARATOR, ".csv", events);
        }
        if(shouldStorePSIFiles){
            final String psiHeader = psiHeaderBegining + numberOfDroplets + psiHeaderEnd;
            writeDataFile(psiHeader, PSISEPARATOR, ".psi", events);
        }
    }
    
    private void writeDataFile(String header, String separator, String fileExtension, CLEvent... events) {
        if(step % stepDumpThreshold != 0) {
            return;
        }                
        File resultFile = new File(directoryName, "result" + step + fileExtension);
        try (FileWriter writer = new FileWriter(resultFile)) {
            Pointer<Float> positionsOut = positions.read(queue, events);
            Pointer<Float> velocitiesOut = velocities.read(queue, events);
            Pointer<Integer> typesOut = types.read(queue, events);
            writer.write(header);
            for (int i = 0; i < numberOfDroplets; i++) {
                writer.write(
                        positionsOut.get(i * VECTOR_SIZE) + separator
                        + positionsOut.get(i * VECTOR_SIZE + 1) + separator
                        + positionsOut.get(i * VECTOR_SIZE + 2) + separator
                        + velocitiesOut.get(i * VECTOR_SIZE) + separator
                        + velocitiesOut.get(i * VECTOR_SIZE + 1) + separator
                        + velocitiesOut.get(i * VECTOR_SIZE + 2) + separator
                        + typesOut.get(i) + "\n");
            }
            positionsOut.release();
            velocitiesOut.release();
            typesOut.release();
        } catch (Exception e) {
            e.printStackTrace();
        }
        
    }    
    
    private void printVelocityProfile(CLEvent... events) {
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
            System.out.println(String.format(Locale.GERMANY, "%e", meanVels[i]));
        }
                
        positionsPointer.release();
        velocitiesPointer.release();
    }
}
