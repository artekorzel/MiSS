package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.*;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Test;

public class TestSimulation implements Simulation {

    private CLContext context;

    public void run() throws Exception {
        context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();

        int dataLength = 10;

        CLBuffer<Test.Data> test = context.createBuffer(CLMem.Usage.InputOutput, Test.Data.class, dataLength);
        Pointer<Test.Data> testPointer = Pointer.allocateArray(Test.Data.class, dataLength);
        for(int i = 0; i < dataLength; ++i) {
            testPointer.set(i, new Test.Data().number(i));
        }
        test.write(queue, testPointer, true);

        Pointer<Integer> seedsMemoryData = Pointer.allocateInts(dataLength).order(context.getByteOrder());
        for(int i = 0; i < dataLength; i++){
            seedsMemoryData.set(i, i);
        }
        CLBuffer<Integer> seedsMemory= context.createIntBuffer(CLMem.Usage.InputOutput, seedsMemoryData);
        Pointer<Float> randomsData = Pointer.allocateFloats(dataLength).order(context.getByteOrder());
        CLBuffer<Float> randoms= context.createFloatBuffer(CLMem.Usage.InputOutput, randomsData);

        CLBuffer<Integer> out = context.createIntBuffer(CLMem.Usage.Output, dataLength);

        Test kernels = new Test(context);
        int[] globalSizes = new int[] { dataLength };
        CLEvent randomEvent = kernels.random_number_kernel(queue, seedsMemory, randoms, 10, globalSizes, null);
        CLEvent randomEvent2 = kernels.random_number_kernel(queue, seedsMemory, randoms, 10, globalSizes, null);
        Pointer<Float> outPtr = randoms.read(queue, randomEvent2);

        for (int i = 0; i < dataLength; i++) {
            System.out.println("randoms[" + i + "] = " + outPtr.get(i));
        }
//        CLEvent addEvt = kernels.return_data(queue, test, out, dataLength, globalSizes, null);
//
//        Pointer<Integer> outPtr = out.read(queue, addEvt);
//
//        for (int i = 0; i < dataLength; i++) {
//            System.out.println("out[" + i + "] = " + outPtr.get(i));
//        }
    }
    private Pointer<Float> allocateFloats(long size) {
        return Pointer.allocateFloats(size).order(context.getByteOrder());
    }
}
