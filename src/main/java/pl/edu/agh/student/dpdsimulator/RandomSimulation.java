package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import java.io.IOException;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.RandomCL;

public class RandomSimulation {
    
    public static void main(String[] args) throws IOException {
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();
        
        RandomCL kernel = new RandomCL(context);
        
        CLBuffer<Float> out = context.createFloatBuffer(CLMem.Usage.InputOutput, 152);
        
        CLEvent evt = kernel.generateRandomNumbers(queue, out, 152, 0, new int[]{30}, null);
        final Pointer<Float> read = out.read(queue, evt);
        
        for(int i = 0; i < 152; i++){
            System.out.println(read.get(i));
        }
    }
}
