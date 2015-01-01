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
    public static final int numberOfSteps = 1000;    
    public static final float deltaTime = 0.1f;
    
    public static final float initBoxSize = 10f;
    public static final float initBoxWidth = initBoxSize;
    public static final float baseRadiusIn = 0.8f * initBoxSize;
    
    public static float boxSize;
    public static float boxWidth;
    public static float radiusIn;
    
//    public static final float temperature = 310.0f;
//    public static final float boltzmanConstant = 1f / temperature;
    
    public static final float lambda = 0.63f;
//    public static final float sigma = 0.075f;
    
    public static final float flowVelocity = 0.05f;
    public static final float thermalVelocity = 0.0036f;
    
    public static final float vesselCutoffRadius = 0.8f;
    public static final float bloodCutoffRadius = 0.4f;
    public static final float plasmaCutoffRadius = 0.4f;
    
//    public static final float vesselDensity = 10000.0f;
//    public static final float bloodDensity = 50000.0f;
//    public static final float plasmaDensity = 50000.0f;
    
    public static final float vesselMass = 1000f;
    public static final float bloodCellMass = 1.14f;
    public static final float plasmaMass = 1f;
    
    public static final float cellRadius = 0.8f;
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
        droplets.set(0, createDropletParameter(vesselMass, lambda));
        droplets.set(1, createDropletParameter(bloodCellMass, lambda));
        droplets.set(2, createDropletParameter(plasmaMass, lambda));
        
        Pointer<Dpd.PairParameters> pairs = parameters.pairs();
        
        Dpd.PairParameters vesselVessel = createPairParameter(
                vesselCutoffRadius, 
                0.0075f,//calculateRepulsionParameter(vesselDensity), 
                0.075f,//sigma, 
                0.0028125f);//calculateGamma(sigma));
        
        Dpd.PairParameters bloodBlood = createPairParameter(
                bloodCutoffRadius, 
                0.0015f,//calculateRepulsionParameter(bloodDensity), 
                0.075f,//sigma, 
                0.0028125f);//calculateGamma(sigma));
        
        Dpd.PairParameters plasmaPlasma = createPairParameter(
                plasmaCutoffRadius, 
                0.0015f,//calculateRepulsionParameter(plasmaDensity), 
                0.075f,//sigma, 
                0.0028125f);//calculateGamma(sigma));
        
        Dpd.PairParameters vesselBlood = createPairParameter(
                vesselCutoffRadius, 
                0.0035f,//(float) Math.sqrt(calculateRepulsionParameter(vesselDensity) * calculateRepulsionParameter(bloodDensity)), 
                0.075f,//sigma, 
                0.0028125f);//calculateGamma(sigma));
        
        Dpd.PairParameters vesselPlasma = createPairParameter(
                vesselCutoffRadius, 
                0.0035f,//(float) Math.sqrt(calculateRepulsionParameter(vesselDensity) * calculateRepulsionParameter(plasmaDensity)), 
                0.075f,//sigma, 
                0.0028125f);//calculateGamma(sigma));
        
        Dpd.PairParameters bloodPlasma = createPairParameter(
                bloodCutoffRadius, 
                0.0075f,//(float) Math.sqrt(calculateRepulsionParameter(bloodDensity) * calculateRepulsionParameter(plasmaDensity)), 
                0.075f,//sigma, 
                0.0028125f);//calculateGamma(sigma));
                        
        pairs.set(0, vesselVessel);        
        pairs.set(1, vesselBlood);
        pairs.set(2, vesselPlasma);
        pairs.set(3, vesselBlood);
        pairs.set(4, bloodBlood);
        pairs.set(5, bloodPlasma);
        pairs.set(6, vesselPlasma);
        pairs.set(7, bloodPlasma);
        pairs.set(8, plasmaPlasma);

        return Arrays.asList(parameters);
    }

    protected Dpd.DropletParameters createDropletParameter(float mass, float lambda) {
        Dpd.DropletParameters dropletParameter = new Dpd.DropletParameters();
        dropletParameter.mass(mass);
        dropletParameter.lambda(lambda);
        return dropletParameter;
    }
    
    protected Dpd.PairParameters createPairParameter(float cutoffRadius, float repulsionParameter, float sigma, float gamma) {
        Dpd.PairParameters pairParameter = new Dpd.PairParameters();
        pairParameter.cutoffRadius(cutoffRadius);
        pairParameter.repulsionParameter(repulsionParameter);
        pairParameter.sigma(sigma);
        pairParameter.gamma(gamma);
        return pairParameter;        
    }
    
//    private float calculateRepulsionParameter(float density) {
//        return 75.0f * boltzmanConstant * temperature / density;
//    }
//    
//    private float calculateGamma(float sigma) {
//        return sigma * sigma / 2.0f / boltzmanConstant / temperature;
//    }
}
