package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.JavaCL;
import java.util.Arrays;


public class ReadParams {
    
    public static void main(String[] args) {
        for (CLPlatform platform : JavaCL.listPlatforms()) {
            CLDevice device = platform.getBestDevice();
            System.out.println(device);
            System.out.println("CL_DEVICE_MAX_COMPUTE_UNITS: " + device.getMaxComputeUnits());
            System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE: " + device.getMaxWorkGroupSize());
            System.out.println("CL_DEVICE_MAX_WORK_ITEM_SIZES: " + Arrays.toString(device.getMaxWorkItemSizes()));
            System.out.println("CL_DEVICE_LOCAL_MEM_SIZE: " + device.getLocalMemSize());
            System.out.println("CL_DEVICE_GLOBAL_MEM_SIZE: " + device.getGlobalMemSize());
            System.out.println();
        }                
    }

}
