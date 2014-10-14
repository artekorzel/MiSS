package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import org.bridj.Pointer;

import pl.edu.agh.student.dpdsimulator.kernels.Dpd;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd.DropletParameter;

public class DpdSimulation {

    private static final int VECTOR_SIZE = 4;

    private static final int numberOfSteps = 10;
    private static final int numberOfDroplets = 50000;
    private static final float deltaTime = 1.0f;
    
    private static final float boxSize = 10.0f;
    private static final float radiusIn = 0.5f * boxSize;
    private static final float radiusOut = 0.75f * boxSize;
    
    private static final float temperature = 310.0f;
    private static final float boltzmanConstant = 1f / temperature / 500f;
    
    private static final float lambda = 0.63f;
    private static final float sigma = 0.075f;
    private static final float gamma = sigma * sigma / 2.0f / boltzmanConstant / temperature;
    
    private static final float flowVelocity = 0.05f;
    private static final float thermalVelocity = 0.0036f;
    
    private static final float vesselCutoffRadius = 0.8f;
    private static final float bloodCutoffRadius = 0.4f;
    private static final float plasmaCutoffRadius = 0.4f;
    
    private static final float vesselDensity = 10000.0f;
    private static final float bloodDensity = 50000.0f;
    private static final float plasmaDensity = 50000.0f;
    
    private static final float vesselMass = 1000f;
    private static final float bloodCellMass = 1.14f;
    private static final float plasmaMass = 1f;

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
    private CLBuffer<Integer> states;
    private CLBuffer<DropletParameter> dropletParameters;
    private CLContext context;
    private CLQueue queue;
    private Dpd dpdKernel;
    private int[] globalSizes;
    private Pointer<Integer> typesPointer;
    private Pointer<Float> partialSumsPointer;
    private Pointer<Float> averageVelocityPointer;
    
    private List<DropletParameter> parameters = new ArrayList<>();
    private String directoryName;
    private int step;
    private int initialRandom;

    /**
     * Inicjalizuje dane oraz wykonuje symulacje przeplywu czastek krwi
     */
    public static void main(String[] args) {
        try {
            DpdSimulation simulation = new DpdSimulation();
            simulation.initData();
            simulation.performSimulation();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Alokuje pamiec na karcie graficznej dla potrzebnych struktur oraz tworzy obiekt kernela, 
     * dzieki ktoremu mozemy wykonywac operacje na karcie graficznej.
     */
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
        partialSumsPointer = Pointer.allocateArray(Float.class, numberOfDroplets * VECTOR_SIZE).order(context.getByteOrder());
        averageVelocity = context.createFloatBuffer(CLMem.Usage.InputOutput, VECTOR_SIZE);
        averageVelocityPointer = Pointer.allocateArray(Float.class, VECTOR_SIZE).order(context.getByteOrder());
        types = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);
        states = context.createIntBuffer(CLMem.Usage.InputOutput, numberOfDroplets);

        dpdKernel = new Dpd(context);
        globalSizes = new int[]{numberOfDroplets};
        directoryName = "../results_" + new Date().getTime();
        File directory = new File(directoryName);
        if(!directory.exists()) {
            directory.mkdir();
        }
    }    
    
    /**
     * Wykonuje zdefiniowana przez nas ilosc krokow symulacji w kazdym kroku zapisujac dane do odpowiednich plikow.
     */
    private void performSimulation() {
        step = 0;
        initDropletParameters();
        initStates();
        initPositionsAndVelocities();
        writePositionsFile(positions);
        initialRandom = random.nextInt();
        CLEvent loopEndEvent = null;
        for (step = 1; step <= numberOfSteps; ++step) {
            System.out.println("\nStep: " + step);
            loopEndEvent = performSingleStep(loopEndEvent);
            printAverageVelocity(loopEndEvent);
            writePositionsFile(newPositions, loopEndEvent);
            swapPositions(loopEndEvent);
            swapVelocities(loopEndEvent);
        }
    }

    /**
     * Tworzy parametry dla trzech typow czasteczek wykorzystywanych w symulacji 
     * odpowiednio dla czastek sciany naczynia, czerwonych krwinek oraz osocza.
     */
    private void initDropletParameters() {
        addParameter(vesselCutoffRadius, vesselMass, vesselDensity, lambda, sigma, gamma);
        addParameter(bloodCutoffRadius, bloodCellMass, bloodDensity, lambda, sigma, gamma);
        addParameter(plasmaCutoffRadius, plasmaMass, plasmaDensity, lambda, sigma, gamma);

        long size = parameters.size();
        Pointer<DropletParameter> valuesPointer = Pointer.allocateArray(DropletParameter.class, size).order(context.getByteOrder());
        for (int i = 0; i < size; i++) {
            valuesPointer.set(i, parameters.get(i));
        }

        dropletParameters = context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }

    /**
     * Dodaje kolejny typ do tablicy parametrow przetrzymywanej na karcie graficznej
     */
    private void addParameter(float cutoffRadius, float mass, float density,
            float lambda, float sigma, float gamma) {
        DropletParameter dropletParameter = new DropletParameter();
        dropletParameter.cutoffRadius(cutoffRadius);
        dropletParameter.mass(mass);
        
        float repulsionParameter = 75.0f * boltzmanConstant * temperature / density;
        dropletParameter.repulsionParameter(repulsionParameter);
        dropletParameter.lambda(lambda);
        dropletParameter.sigma(sigma);
        dropletParameter.gamma(gamma);

        parameters.add(dropletParameter);
    }

    private void initStates(){
        Pointer<Integer> statesPointer = Pointer.allocateArray(Integer.class, numberOfDroplets).order(context.getByteOrder());
        for(int i = 0; i < numberOfDroplets; i++){
            statesPointer.set(i,random.nextInt(2147483647));
        }
        
        states = context.createBuffer(CLMem.Usage.InputOutput, statesPointer);
    }
    
    /**
     * Generuje naczynie krwionosne wraz z czasteczkami znajdujacymi sie wewnatrz oraz poczatkowe predkosci czastek.
     */
//    private void initPositionsAndVelocities() {
//        long t = System.nanoTime();
//        generateTube();
//        generateVelocities();
//        System.out.println("Initialization time: " + (System.nanoTime() - t));
//    }
    
    private CLEvent initPositionsAndVelocities() {
        CLEvent generatePositionsEvent = dpdKernel.generateTube(queue, positions, types, states, numberOfDroplets, 
                0.4f * boxSize, 0.5f * boxSize, boxSize, globalSizes, null);
        return dpdKernel.generateRandomVector(queue, velocities, states, types, thermalVelocity, flowVelocity, numberOfDroplets,
                globalSizes, null, generatePositionsEvent);
    }
    /**
     * Generuje naczynie krwionosne oraz polozenia czasteczek znajdujacych sie w jego wnetrzu. Sciana naczynia zostaje
     * kazda czastka w okreslonej odleglosci od srodka natomiast pozostale sa losowo wybierane jako osocze lub krwinka.
     */
//    private void generateTube() {
//        int numberOfCoordinates = numberOfDroplets * VECTOR_SIZE;
//        typesPointer = Pointer.allocateArray(Integer.class, numberOfDroplets).order(context.getByteOrder());
//        Pointer<Float> positionsPointer = Pointer.allocateArray(Float.class, numberOfCoordinates).order(context.getByteOrder());
//        for (int i = 0; i < numberOfCoordinates; i += VECTOR_SIZE) {
//            float x = nextRandomFloat(radiusOut);
//            float z = nextRandomFloat((float) Math.sqrt(radiusOut * radiusOut - x * x));
//            float y = nextRandomFloat(boxSize);
//
//            positionsPointer.set(i, x);
//            positionsPointer.set(i + 1, y);
//            positionsPointer.set(i + 2, z);
//            positionsPointer.set(i + 3, 0.0f);
//
//            float distanceFromY = (float) Math.sqrt(x * x + z * z);
//            int dropletId = i / 4;
//            if (distanceFromY >= radiusIn) {
//                typesPointer.set(dropletId, 0);
//            } else {
//                float randomNum = nextRandomFloat(1);
//                if (randomNum >= 0.0f) {
//                    typesPointer.set(dropletId, 1);
//                } else {
//                    typesPointer.set(dropletId, 2);
//                }
//            }
//        }
//        positions = context.createBuffer(CLMem.Usage.InputOutput, positionsPointer);
//        types = context.createBuffer(CLMem.Usage.InputOutput, typesPointer);
//    }

    /**
     * Generuje predkosci poczatkowe czasteczek. Czasteczki scian pozostaja nieruchome, zas czasteczki osocza i krwinki
     * w kierunku zgodnym z przeplywem krwi maja predkosci losowane z przedzialu <0; velocityInitRange> oraz
     * <-thermalVelocity, thermalVelocity> w pozostalych kierunkach
     */
//    private void generateVelocities() {
//        int numberOfCoordinates = numberOfDroplets * VECTOR_SIZE;
//        Pointer<Float> velocitiesPointer = Pointer.allocateArray(Float.class, numberOfCoordinates).order(context.getByteOrder());
//        for (int i = 0; i < numberOfCoordinates; i += VECTOR_SIZE) {
//            int dropletId = i / 4;
//            float x, y, z;
//            if(typesPointer.get(dropletId) == 0) {
//                x = 0.0f;
//                y = 0.0f;
//                z = 0.0f;
//            } else {
//                x = nextRandomFloat(thermalVelocity);
//                y = (nextRandomFloat(flowVelocity) + flowVelocity) / 2.0f;
//                z = nextRandomFloat(thermalVelocity);
//            }
//
//            velocitiesPointer.set(i, x);
//            velocitiesPointer.set(i + 1, y);
//            velocitiesPointer.set(i + 2, z);
//            velocitiesPointer.set(i + 3, 0.0f);
//        }
//        velocities = context.createBuffer(CLMem.Usage.InputOutput, velocitiesPointer);
//    }

    /**
     * Generuje zmienna losowa z przedzialu <-range; range>
     */
    private float nextRandomFloat(float range) {
        return random.nextFloat() * 2 * range - range;
    }

    /**
     * Wykonuje pojedynczy krok symulacji
     */
    private CLEvent performSingleStep(CLEvent previousStepEvent) {
        CLEvent forcesEvent = calculateForces(previousStepEvent);
        CLEvent newPositionsAndPredictedVelocitiesEvent = 
                calculateNewPositionsAndPredictedVelocities(forcesEvent);
        return calculateNewVelocities(newPositionsAndPredictedVelocitiesEvent);
    }

    /**
     * Wywoluje wykonanie obliczenia sil na karcie graficznej
     */
    private CLEvent calculateForces(CLEvent gaussianRandomsEvent) {
        return dpdKernel.calculateForces(queue, positions, velocities, forces, dropletParameters, types,
                numberOfDroplets, step + initialRandom, globalSizes, null, gaussianRandomsEvent);
    }

    /**
     * Wywoluje wykonanie obliczenia nowych pozycji i przewidywanych predkosci na karcie graficznej
     */
    private CLEvent calculateNewPositionsAndPredictedVelocities(CLEvent forcesEvent) {
        return dpdKernel.calculateNewPositionsAndPredictedVelocities(queue, positions, velocities, forces, newPositions,
                predictedVelocities, dropletParameters, types, deltaTime, numberOfDroplets, boxSize, globalSizes, null, forcesEvent);
    }

    /**
     * Wywoluje wykonanie obliczenia nowych predkosci na karcie graficznej
     */
    private CLEvent calculateNewVelocities(CLEvent newPositionsAndPredictedVelocitiesEvent) {
        return dpdKernel.calculateNewVelocities(queue, newPositions, velocities, predictedVelocities, newVelocities,
                forces, dropletParameters, types, deltaTime, numberOfDroplets, step + initialRandom,
                globalSizes, null, newPositionsAndPredictedVelocitiesEvent);
    }

    /**
     * Zamienia wskazniki polozen przechowywanych na karcie graficznej
     */
    private void swapPositions(CLEvent... events) {
        CLEvent.waitFor(events);
        CLBuffer<Float> tmp = positions;
        positions = newPositions;
        newPositions = tmp;
    }

    /**
     * Zamienia wskazniki predkosci przechowywanych na karcie graficznej
     */
    private void swapVelocities(CLEvent... events) {
        CLEvent.waitFor(events);
        CLBuffer<Float> tmp = velocities;
        velocities = newVelocities;
        newVelocities = tmp;
    }

    /**
     * Zapisuje dane czastek do plików ..\results_<timestamp>\result<numer_kroku>.
     * W kolejnych liniach dla kazdej czastki znajduja sie odpowiednio wspolrzedne polozenia x, y, z oraz typ:
     * 0 - sciana, 1 - krwinka, 2 - osocze
     */
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

    /**
     * Oblicza i wypisuje informacje o predkosci sredniej wszystkich czasteczek
     */
    private void printAverageVelocity(CLEvent... events) {
        for(int i = 0; i < VECTOR_SIZE; ++i) {
            averageVelocityPointer.set(i, 0.0f);
        }
        
        for(int i = 0; i < numberOfDroplets * VECTOR_SIZE; ++i) {
            partialSumsPointer.set(i, 0.0f);
        }
        
        CLEvent fillBuffers = averageVelocity.write(queue, averageVelocityPointer, true, events);
        fillBuffers = partialSums.write(queue, partialSumsPointer, true, fillBuffers);
        CLEvent reductionEvent = dpdKernel.doVectorReduction(queue, newVelocities, partialSums,
                averageVelocity, numberOfDroplets, globalSizes, null, fillBuffers);
        Pointer<Float> out = averageVelocity.read(queue, reductionEvent);
        System.out.println("avgVel = (" + out.get(0) + ", " + out.get(1) + ", " + out.get(2) + ")");
        out.release();
    }
}
