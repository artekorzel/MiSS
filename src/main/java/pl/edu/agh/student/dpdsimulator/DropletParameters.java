package pl.edu.agh.student.dpdsimulator;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLMem;
import java.util.ArrayList;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd.DropletParameter;

public class DropletParameters {
    static ArrayList<DropletParameter> parameters = new ArrayList<>();
    
    public static void addParameter(float temperature, float density, float repulsionParameter,
                float lambda, float sigma, float gamma){
        DropletParameter d = new DropletParameter();
        d.temperature(temperature);
        d.density(density);
        d.repulsionParameter(repulsionParameter);
        d.lambda(lambda);
        d.sigma(sigma);
        d.gamma(gamma);
        
        parameters.add(d);
    }
    
    public static CLBuffer<DropletParameter> buildBuffer(CLContext context){
        long size = parameters.size();
        Pointer<DropletParameter> valuesPointer = Pointer.allocateArray(DropletParameter.class, size).order(context.getByteOrder());        
        for(int i = 0; i < size; i++){
            valuesPointer.set(i, parameters.get(i));
        }
        return context.createBuffer(CLMem.Usage.InputOutput, valuesPointer);
    }                                                 
}   
