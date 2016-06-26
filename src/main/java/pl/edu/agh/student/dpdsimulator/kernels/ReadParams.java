package pl.edu.agh.student.dpdsimulator.kernels;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLQueue;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;


public class ReadParams {
    
    static CLKernel kernel;
    
    public static void main(String[] args) throws IOException {
        for (CLPlatform platform : JavaCL.listPlatforms()) {
            CLDevice device = platform.getBestDevice();
            System.out.println(device);
            System.out.println("CL_DEVICE_MAX_COMPUTE_UNITS: " + device.getMaxComputeUnits());
            System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE: " + device.getMaxWorkGroupSize());
            System.out.println("CL_DEVICE_MAX_WORK_ITEM_SIZES: " + Arrays.toString(device.getMaxWorkItemSizes()));
            System.out.println("CL_DEVICE_LOCAL_MEM_SIZE: " + device.getLocalMemSize());
            System.out.println("CL_DEVICE_GLOBAL_MEM_SIZE: " + device.getGlobalMemSize());            
            System.out.println("CL_DEVICE_MAX_MEM_ALLOC_SIZE: " + device.getMaxMemAllocSize());       
            
            CLContext context = JavaCL.createContext(null, device);
            new Dpd(context){
                {
                ReadParams.kernel = createKernel("generateRandom");
                }
            };
            Map<CLDevice, Long> preferredWorkGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple();
            for(Map.Entry<CLDevice, Long> e : preferredWorkGroupSizeMultiple.entrySet()) {
                System.out.println("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " + e.getValue());
            } 
            System.out.println();
        }                
    }

}
