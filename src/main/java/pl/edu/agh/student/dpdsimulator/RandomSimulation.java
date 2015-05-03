package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.LocalSize;
import java.io.IOException;
import java.util.Arrays;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.RandomCL;

public class RandomSimulation {
    
    public static void main(String[] args) throws IOException {
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();
        
        RandomCL kernel = new RandomCL(context);
        
        CLBuffer<Float> out = context.createFloatBuffer(CLMem.Usage.InputOutput, 100);
        
        CLEvent evt = kernel.random(queue, out, 100, new int[]{100}, null);
        final Pointer<Float> read = out.read(queue, evt);
        
        for(int i = 0; i < 100; i++){
            System.out.println(read.get(i));
        }
    }
}
