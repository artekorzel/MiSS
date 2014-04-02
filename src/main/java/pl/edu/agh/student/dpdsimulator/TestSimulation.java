package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Test;

public class TestSimulation implements Simulation {

    public void run() throws Exception {
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();

        int dataLength = 10;

        CLBuffer<Test.Data> test = context.createBuffer(CLMem.Usage.InputOutput, Test.Data.class, dataLength);
        Pointer<Test.Data> testPointer = Pointer.allocateArray(Test.Data.class, dataLength);
        for(int i = 0; i < dataLength; ++i) {
            testPointer.set(i, new Test.Data().number(i));
        }
        test.write(queue, testPointer, true);

        CLBuffer<Integer> out = context.createIntBuffer(CLMem.Usage.Output, dataLength);

        Test kernels = new Test(context);
        int[] globalSizes = new int[] { dataLength };
        CLEvent addEvt = kernels.return_data(queue, test, out, dataLength, globalSizes, null);

        Pointer<Integer> outPtr = out.read(queue, addEvt);

        for (int i = 0; i < dataLength; i++) {
            System.out.println("out[" + i + "] = " + outPtr.get(i));
        }
    }
}
