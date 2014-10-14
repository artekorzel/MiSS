/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import java.io.File;
import java.io.FileWriter;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.*;
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
        globalSizes = new int[]{STEPS};        
        String directoryName = "../rands";
        File directory = new File(directoryName);
        if(!directory.exists()) {
            directory.mkdir();
        }
                
        Pointer<Float> positionsPointer = Pointer.allocateFloats(STEPS).order(context.getByteOrder());        
        for(int i = 0; i < STEPS; i++){
            positionsPointer.set(i, 0.0f);
        }
        rands = context.createBuffer(CLMem.Usage.InputOutput, positionsPointer, false);
        
        CLEvent randomEvent = randomKernel.random(queue, rands, STEPS, globalSizes, null, null);
        File resultFile = new File(directoryName, "rands.csv");
        try (FileWriter writer = new FileWriter(resultFile)) {
            Pointer<Float> out = rands.read(queue);    
            for(int i = 0; i < STEPS; i++){
                writer.write(i + ":" + out.get(i) + "\n");
            }
            out.release();
        }
    }
}
