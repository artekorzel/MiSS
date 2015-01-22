/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.RandomCL;
import pl.edu.agh.student.dpdsimulator.kernels.RandomCL.DropletParameters;
import pl.edu.agh.student.dpdsimulator.kernels.RandomCL.PairParameters;
import pl.edu.agh.student.dpdsimulator.kernels.RandomCL.Parameters;
/**
 *
 * @author Filip
 */
public class RandomSimulation {
    int STEPS = 1000000;
    
    private  CLBuffer<Float> rands;
    private int[] globalSizes;
    
    public static void main(String[] args) {
        try {
            RandomSimulation rsim = new RandomSimulation();
            rsim.randomizuj();
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void randomizuj() throws Exception{
        CLContext context = JavaCL.createBestContext();             
        CLQueue queue = context.createDefaultQueue();
        RandomCL randomKernel = new RandomCL(context);           
        globalSizes = new int[]{1};        
//        String directoryName = "../rands";
//        File directory = new File(directoryName);
//        if(!directory.exists()) {
//            directory.mkdir();
//        }
                
        List<Parameters> params = createParameters();
        long size = params.size();
        Pointer<Parameters> valuesPointer
                = Pointer.allocateArray(Parameters.class, size).order(context.getByteOrder());
        for (int i = 0; i < size; i++) {
            valuesPointer.set(i, params.get(i));
        }

        CLBuffer<Parameters> parameters = context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
        CLBuffer<Float> floats = context.createFloatBuffer(CLMem.Usage.InputOutput, 9);
        
        CLEvent randomEvent = randomKernel.test(queue, parameters, floats, globalSizes, null);
        
        
        Pointer<Float> testFOut = floats.read(queue, randomEvent);
        for(int i = 0; i < 9; ++i) {
            System.out.println(testFOut.get(i));
        }
        testFOut.release();
    }
    
    public static final int VECTOR_SIZE = 4;
    public static final double NANOS_IN_SECOND = 1000000000.0;
    
    public static final boolean shouldStoreFiles = false;
    
    public static final int numberOfCellNeighbours = 27;
    public static final int numberOfSteps = 300;    
    public static final float deltaTime = 0.1f;
    
    public static final float initBoxSize = 10f;
    public static final float initBoxWidth = initBoxSize;
    public static final float baseRadiusIn = 0.8f * initBoxSize;
    
    public static float boxSize;
    public static float boxWidth;
    public static float radiusIn;
    
    public static final float lambda = 0.63f;
    
    public static final float flowVelocity = 0.05f;
    public static final float thermalVelocity = 0.0036f;
    
    public static final float vesselCutoffRadius = 0.8f;
    public static final float bloodCutoffRadius = 0.4f;
    public static final float plasmaCutoffRadius = 0.4f;
    
    public static final float vesselMass = 1000f;
    public static final float bloodCellMass = 1.14f;
    public static final float plasmaMass = 1f;
    
    public static final float cellRadius = 0.8f;
    public static final int baseNumberOfCells = (int) (Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxWidth / cellRadius));
    public static final int baseNumberOfDroplets = baseNumberOfCells * 4;
    
    public static int numberOfCells;
    public static int numberOfDroplets;
    public static final int maxDropletsPerCell = (baseNumberOfDroplets / baseNumberOfCells) * 5 + 1;
        
    protected List<Parameters> createParameters() {
        Parameters parameters = new Parameters();
        Pointer<DropletParameters> droplets = parameters.droplets();
        droplets.set(0, createDropletParameter(vesselMass, lambda));
        droplets.set(1, createDropletParameter(bloodCellMass, lambda));
        droplets.set(2, createDropletParameter(plasmaMass, lambda));
        
        Pointer<PairParameters> pairs = parameters.pairs();
        
        PairParameters vesselVessel = createPairParameter(vesselCutoffRadius, 
                0.0075f,
                0.0028f, 
                0.002f);
        
        PairParameters bloodBlood = createPairParameter(bloodCutoffRadius, 
                0.0015f,
                0.0028f, 
                0.001f);
        
        PairParameters plasmaPlasma = createPairParameter(plasmaCutoffRadius, 
                0.0015f,
                0.028f, 
                0.002f);
        
        PairParameters vesselBlood = createPairParameter(vesselCutoffRadius, 
                0.0035f,
                0.0028f, 
                0.0007f);
        
        PairParameters vesselPlasma = createPairParameter(vesselCutoffRadius, 
                0.0035f, 
                0.0028f, 
                0.002f);
        
        PairParameters bloodPlasma = createPairParameter(bloodCutoffRadius, 
                0.0075f, 
                0.0028f, 
                0.004f);
                        
        pairs.set(0, vesselVessel);        
        pairs.set(1, vesselBlood);
        pairs.set(2, vesselPlasma);
        pairs.set(3, vesselBlood);
        pairs.set(4, bloodBlood);
        pairs.set(5, bloodPlasma);
        pairs.set(6, vesselPlasma);
        pairs.set(7, bloodPlasma);
        pairs.set(8, plasmaPlasma);

        return Arrays.asList(parameters);
    }

    protected DropletParameters createDropletParameter(float mass, float lambda) {
        DropletParameters dropletParameter = new DropletParameters();
        dropletParameter.mass(mass);
        dropletParameter.lambda(lambda);
        return dropletParameter;
    }
    
    protected PairParameters createPairParameter(float cutoffRadius, float pi, float gamma, float sigma) {
        PairParameters pairParameter = new PairParameters();
        pairParameter.cutoffRadius(cutoffRadius);
        pairParameter.pi(pi);
        pairParameter.sigma(sigma);
        pairParameter.gamma(gamma);
        return pairParameter;        
    }
}
