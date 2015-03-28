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
        
        float[] aaa = new float[]{1,2,3,0,3,1,2,0,2,3,1,0};
        CLBuffer<Float> in = context.createFloatBuffer(CLMem.Usage.InputOutput, Pointer.pointerToFloats(aaa));
        CLBuffer<Float> out = context.createFloatBuffer(CLMem.Usage.InputOutput, 4);
        
        CLEvent evt = kernel.reduction(queue, in, LocalSize.ofFloatArray(4), out, 12, new int[]{8}, new int[]{4});
        final Pointer<Float> read = out.read(queue, evt);
        
        System.out.println(Arrays.toString(read.getFloats()));
    }
}
