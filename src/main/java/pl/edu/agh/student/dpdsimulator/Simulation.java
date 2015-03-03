package pl.edu.agh.student.dpdsimulator;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.bridj.Pointer;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public abstract class Simulation {

    public static final int VECTOR_SIZE = 4;
    public static final double NANOS_IN_SECOND = 1000000000.0;
    
    public static boolean shouldStoreFiles;
    
    public static int numberOfCellNeighbours = 27;
    public static int numberOfSteps;    
    public static float deltaTime;
    
    public static float initBoxSize = 8f;
    public static float initBoxWidth = initBoxSize;
    
    public static float boxSize;
    public static float boxWidth;
    public static float radiusIn;    
    public static float boxSizeScale;
    public static float boxWidthScale;
        
    public static float cellRadius;
    public static int baseNumberOfCells;
    public static int baseNumberOfDroplets;    
    
    public static int numberOfCells;
    public static int numberOfDroplets;
    public static int maxDropletsPerCell;
        
    public abstract void initData() throws IOException;
    
    public abstract void performSimulation();

//    protected List<Dpd.Parameters> createParameters() {
//        Dpd.Parameters parameters = new Dpd.Parameters();
//        Pointer<Dpd.DropletParameters> droplets = parameters.droplets();
//        droplets.set(0, createDropletParameter(1000f, lambda));
//        droplets.set(1, createDropletParameter(1f, lambda));
//        droplets.set(2, createDropletParameter(1f, lambda));
//        
//        Pointer<Dpd.PairParameters> pairs = parameters.pairs();
//        
//        Dpd.PairParameters vesselVessel = createPairParameter(0.8f,
//                1e-6f,
//                1e-3f,
//                1e-4f);
//        
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
////        for(int i = 0; i < 9; ++i) {
////            pairs.set(i, vesselVessel);
////        }
//
//        return Arrays.asList(parameters);
//    }

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
    
    protected void loadInitialDataFromFile(String filename){
        try{
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

            shouldStoreFiles = getBoolean(bufferedReader.readLine());
            numberOfSteps = getInt(bufferedReader.readLine());
            deltaTime = getFloat(bufferedReader.readLine());
            cellRadius = getFloat(bufferedReader.readLine());
            boxSizeScale = getFloat(bufferedReader.readLine());
            boxWidthScale = getFloat(bufferedReader.readLine());
            
            baseNumberOfCells = (int) (Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxWidth / cellRadius));
            baseNumberOfDroplets = baseNumberOfCells * 4;
            maxDropletsPerCell = (baseNumberOfDroplets / baseNumberOfCells) * 5 + 1;            
            numberOfDroplets = baseNumberOfDroplets / 8;
        } catch (Exception e){
            e.printStackTrace();
        }
    }
    
    protected List<Dpd.Parameters> loadParametersFromFile(String filename){
        Dpd.Parameters parameters = new Dpd.Parameters();        
        int numberOfCellKinds;
        float[] mass, lambda;
        float[][] cutOffRadius, pi, gamma, sigma;
        try {                              
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
            
            numberOfCellKinds = getInt(bufferedReader.readLine());
            
            mass = new float[numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                mass[i] = getFloat(bufferedReader.readLine());
            }
            
            lambda = new float[numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                lambda[i] = getFloat(bufferedReader.readLine());
            }
            
            Pointer<Dpd.DropletParameters> droplets = parameters.droplets();
            for(int i = 0; i < numberOfCellKinds; i++){                                
                droplets.set(i, createDropletParameter(mass[i], lambda[i]));                
            }
            
            cutOffRadius = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    cutOffRadius[i][j] = cutOffRadius[j][i] = getFloat(bufferedReader.readLine());
                }
            }
            
            pi = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    pi[i][j] = pi[j][i] = getFloat(bufferedReader.readLine());
                }
            }
                        
            gamma = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    gamma[i][j] = gamma[j][i] = getFloat(bufferedReader.readLine());
                }
            }
            
            sigma = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    sigma[i][j] = sigma[j][i] = getFloat(bufferedReader.readLine());
                }
            }
            
            int counter = 0;
            Pointer<Dpd.PairParameters> pairs = parameters.pairs();
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = 0; j < numberOfCellKinds; j++){
                    pairs.set(counter++, createPairParameter(cutOffRadius[i][j], pi[i][j], gamma[i][j], sigma[i][j]));
                }
            }                        
        } catch (FileNotFoundException ex) {
            
        } catch (IOException ex) {
            
        }
        return Arrays.asList(parameters);
    }

    private boolean getBoolean(String line) {
        String result = line.split(":")[1];
        return Boolean.parseBoolean(result);
    }
    
    private int getInt(String line) {
        String result = line.split(":")[1];
        return Integer.parseInt(result);
    }
    
    private float getFloat(String line) {
        String result = line.split(":")[1];
        return Float.parseFloat(result);
    }
}
