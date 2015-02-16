package pl.edu.agh.student.dpdsimulator;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public abstract class Simulation {

    public static final int VECTOR_SIZE = 4;
    public static final double NANOS_IN_SECOND = 1000000000.0;
    
    public static final boolean shouldStoreFiles = false;
    
    public static final int numberOfCellNeighbours = 27;
    public static final int numberOfSteps = 10000;    
    public static final float deltaTime = 0.01f;
    
    public static final float initBoxSize = 5f;
    public static final float initBoxWidth = initBoxSize;
    
    public static float boxSize;
    public static float boxWidth;
    
    public static final float lambda = 0.5f;    
    public static final float cutoffRadius = 0.5f;    
    public static final float mass = 1f;
    
    public static final float cellRadius = 0.5f;
    public static final int baseNumberOfCells = (int) (Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxWidth / cellRadius));
    public static final int baseNumberOfDroplets = baseNumberOfCells * 4;
    
    public static int numberOfCells;
    public static int numberOfDroplets;
    public static final int maxDropletsPerCell = (baseNumberOfDroplets / baseNumberOfCells) * 5 + 1;
        
    public abstract void initData(float boxSize, float boxWidth, int numberOfDroplets) throws IOException;
    
    public abstract void performSimulation();

    protected List<Dpd.Parameters> createParameters() {
        Dpd.Parameters parameters = new Dpd.Parameters();
        
        parameters.mass(mass);
        parameters.lambda(lambda);
        parameters.cutoffRadius(cutoffRadius);
        parameters.pi(1e-02f);
        return Arrays.asList(parameters);
    }
}
