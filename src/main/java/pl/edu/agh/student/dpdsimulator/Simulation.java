package pl.edu.agh.student.dpdsimulator;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public abstract class Simulation {

    public static final int VECTOR_SIZE = 4;
    public static final double NANOS_IN_SECOND = 1000000000.0;
    
    public static final boolean shouldStoreFiles = false;
    
    public static final int numberOfCellNeighbours = 27;
    public static final int numberOfSteps = 20000;    
    public static final float deltaTime = 0.5f;
    
    public static final float initBoxSize = 4f;
    public static final float initBoxWidth = initBoxSize;
    
    public static float boxSize;
    public static float boxWidth;
    public static float radiusIn;
    
    public static final float lambda = 0.5f;
        
    public static final float cellRadius = 0.4f;
    public static final int baseNumberOfCells = (int) (Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxWidth / cellRadius));
    public static final int baseNumberOfDroplets = baseNumberOfCells * 4;
    
    public static int numberOfCells;
    public static int numberOfDroplets;
    public static final int maxDropletsPerCell = (baseNumberOfDroplets / baseNumberOfCells) * 5 + 1;
        
    public abstract void initData(float boxSize, float boxWidth, int numberOfDroplets) throws IOException;
    
    public abstract void performSimulation();

    protected List<Dpd.Parameters> createParameters() {
        Dpd.Parameters parameters = new Dpd.Parameters();
        Pointer<Dpd.DropletParameters> droplets = parameters.droplets();
        droplets.set(0, createDropletParameter(1f, lambda));
        droplets.set(1, createDropletParameter(1f, lambda));
        droplets.set(2, createDropletParameter(1f, lambda));
        
        Pointer<Dpd.PairParameters> pairs = parameters.pairs();
        
        Dpd.PairParameters vesselVessel = createPairParameter(0.4f,
                1e-5f,
                -1e-2f,
                1e-3f);
        
//        Dpd.PairParameters bloodBlood = createPairParameter(0.4f,
//                1e-6f,
//                1e-3f,
//                1e-4f);
//        
//        Dpd.PairParameters plasmaPlasma = createPairParameter(0.4f,
//                1e-6f,
//                1e-3f,
//                1e-4f);
//        
//        Dpd.PairParameters vesselBlood = createPairParameter(0.8f,
//                1e-6f,
//                -6e-3f,
//                1e-4f);
//        
//        Dpd.PairParameters vesselPlasma = createPairParameter(0.8f,
//                1e-6f,
//                -6e-3f,
//                1e-4f);
//        
//        Dpd.PairParameters bloodPlasma = createPairParameter(0.4f,
//                1e-6f,
//                1e-3f,
//                1e-4f);
//                        
//        pairs.set(0, vesselVessel);        
//        pairs.set(1, vesselBlood);
//        pairs.set(2, vesselPlasma);
//        pairs.set(3, vesselBlood);
//        pairs.set(4, bloodBlood);
//        pairs.set(5, bloodPlasma);
//        pairs.set(6, vesselPlasma);
//        pairs.set(7, bloodPlasma);
//        pairs.set(8, plasmaPlasma);
        for(int i = 0; i < 9; ++i) {
            pairs.set(i, vesselVessel);
        }

        return Arrays.asList(parameters);
    }

    protected Dpd.DropletParameters createDropletParameter(float mass, float lambda) {
        Dpd.DropletParameters dropletParameter = new Dpd.DropletParameters();
        dropletParameter.mass(mass);
        dropletParameter.lambda(lambda);
        return dropletParameter;
    }
    
    protected Dpd.PairParameters createPairParameter(float cutoffRadius, float pi, float gamma, float sigma) {
        Dpd.PairParameters pairParameter = new Dpd.PairParameters();
        pairParameter.cutoffRadius(cutoffRadius);
        pairParameter.pi(pi);
        pairParameter.sigma(sigma);
        pairParameter.gamma(gamma);
        return pairParameter;        
    }
}
