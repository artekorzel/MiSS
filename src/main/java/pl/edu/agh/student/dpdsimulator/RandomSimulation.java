/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.*;
/**
 *
 * @author Filip
 */
public class RandomSimulation {
    int STEPS = 1;
    
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
        rands = context.createFloatBuffer(CLMem.Usage.InputOutput, STEPS);
        
        globalSizes = new int[]{STEPS};
        
        CLEvent randomEvent = randomKernel.random(queue, rands, STEPS, globalSizes, null, null);
        Pointer<Float> out = rands.read(queue, randomEvent);    
        for(int i = 0; i < STEPS; i++){
            System.out.println(i + ":" + out.get(i));
        }
        out.release();
    }
}
