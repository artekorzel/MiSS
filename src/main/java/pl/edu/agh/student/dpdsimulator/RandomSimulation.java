package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import java.io.IOException;
import java.util.Arrays;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.RandomCL;

public class RandomSimulation {
    
    public static void main(String[] args) throws IOException {
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();
                
        RandomCL kernel = new RandomCL(context);
                
        CLBuffer<Float> res = context.createFloatBuffer(CLMem.Usage.InputOutput, 1);
        
        CLEvent evt = kernel.test2(queue, res, new int[]{1}, null);
        float value = res.read(queue, evt).getFloat();
        
        System.out.println(value);
    }
}
