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
                
        CLBuffer<Integer> res = context.createIntBuffer(CLMem.Usage.InputOutput, 1);
        Pointer<Float> aaaP = Pointer.allocateFloats(4).order(context.getByteOrder());
        aaaP.set(0, -0.4f);
        aaaP.set(1, -0.1f);
        aaaP.set(2, 0.26f);
        CLBuffer<Float> aaa = context.createFloatBuffer(CLMem.Usage.InputOutput, aaaP);
        
        CLEvent evt = kernel.calculateCellId(queue, res, aaa, 0.5f, 0.75f, 1.0f, new int[]{1}, null);
        final int cellId = res.read(queue, evt).getInt();
        
        System.out.println(cellId);
                
        res = context.createIntBuffer(CLMem.Usage.InputOutput, 3);
        
        evt = kernel.calculateCellCoordinates(queue, res, cellId, 0.5f, 0.75f, 1.0f, 3, 4, new int[]{1}, null);
        System.out.println(Arrays.toString(res.read(queue, evt).getInts()));
    }
}
