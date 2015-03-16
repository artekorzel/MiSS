package pl.edu.agh.student.dpdsimulator;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
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
    
    public static float initBoxSize;
    public static float initBoxWidth;
    
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
    public static float numberOfDropletsFactor;
    public static int maxDropletsPerCell;
    public static int averageDropletsPerCell;
    public static int numberOfCellKinds;
    
    protected Pointer<Dpd.DropletParameters> dropletParametersPointer;
    protected Pointer<Dpd.PairParameters> pairParametersPointer;
        
    public abstract void initData() throws IOException;
    
    public abstract void performSimulation();

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
            Properties prop = getProperties();

            shouldStoreFiles = Boolean.parseBoolean(prop.getProperty("shouldStoreFiles"));
            numberOfSteps = Integer.parseInt(prop.getProperty("numberOfSteps"));
            deltaTime = Float.parseFloat(prop.getProperty("deltaTime"));
            cellRadius = Float.parseFloat(prop.getProperty("cellRadius"));
            initBoxSize = Float.parseFloat(prop.getProperty("initBoxSize"));
            initBoxWidth = Float.parseFloat(prop.getProperty("initBoxWidth"));
            boxSizeScale = Float.parseFloat(prop.getProperty("boxSizeScale"));
            boxWidthScale = Float.parseFloat(prop.getProperty("boxWidthScale"));
            maxDropletsPerCell = Integer.parseInt(prop.getProperty("maxDropletsPerCell"));
            averageDropletsPerCell = Integer.parseInt(prop.getProperty("averageDropletsPerCell"));
            numberOfDroplets = Integer.parseInt(prop.getProperty("numberOfDroplets"));
            
            baseNumberOfCells = (int) (Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxSize / cellRadius) * Math.ceil(2 * initBoxWidth / cellRadius));
            baseNumberOfDroplets = baseNumberOfCells * averageDropletsPerCell;
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    protected void loadParametersFromFile(String filename){        
        float[] mass, lambda;
        float[][] cutOffRadius, pi, gamma, sigma;
        try {                              
            Properties prop = getProperties();
            
            numberOfCellKinds = Integer.parseInt(prop.getProperty("numberOfCellKinds"));
            
            mass = new float[numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                mass[i] = Float.parseFloat(prop.getProperty("mass(" + i + ")"));
            }
            
            lambda = new float[numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                lambda[i] = Float.parseFloat(prop.getProperty("lambda(" + i + ")"));
            }
                        
            dropletParametersPointer = Pointer.allocateArray(Dpd.DropletParameters.class, numberOfCellKinds);
            for(int i = 0; i < numberOfCellKinds; i++){                                
                dropletParametersPointer.set(i, createDropletParameter(mass[i], lambda[i]));                
            }
            
            
            
            cutOffRadius = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    cutOffRadius[i][j] = cutOffRadius[j][i] = Float.parseFloat(prop.getProperty("cutoffRadius(" + i + "," + j + ")"));
                }
            }
            
            pi = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    pi[i][j] = pi[j][i] = Float.parseFloat(prop.getProperty("pi(" + i + "," + j + ")"));
                }
            }
                        
            gamma = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    gamma[i][j] = gamma[j][i] = Float.parseFloat(prop.getProperty("gamma(" + i + "," + j + ")"));
                }
            }
            
            sigma = new float[numberOfCellKinds][numberOfCellKinds];
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = i; j < numberOfCellKinds; j++){
                    sigma[i][j] = sigma[j][i] = Float.parseFloat(prop.getProperty("sigma(" + i + "," + j + ")"));
                }
            }
            
            int counter = 0;            
            pairParametersPointer = Pointer.allocateArray(Dpd.PairParameters.class, numberOfCellKinds * numberOfCellKinds);
            for(int i = 0; i < numberOfCellKinds; i++){
                for(int j = 0; j < numberOfCellKinds; j++){
                    pairParametersPointer.set(counter++, createPairParameter(cutOffRadius[i][j], pi[i][j], gamma[i][j], sigma[i][j]));
                }
            }                        
        } catch (FileNotFoundException ex) {
            
        } catch (IOException ex) {
            
        }
    }

    private Properties getProperties() throws IOException {
        Properties prop = new Properties();
        String propFileName = "simulation.data";
        InputStream inputStream = new FileInputStream(propFileName);
        if (inputStream != null) {
            prop.load(inputStream);
        } else {
            throw new FileNotFoundException("property file '" + propFileName + "' not found in the classpath");
        }
        return prop;
    }
    
    private boolean getBoolean(String line) {
        String result = line.split(":")[1].replaceAll("\\s", "");
        return Boolean.parseBoolean(result);
    }
    
    private int getInt(String line) {
        String result = line.split(":")[1].replaceAll("\\s", "");
        return Integer.parseInt(result);
    }
    
    private float getFloat(String line) {
        String result = line.split(":")[1].replaceAll("\\s", "");
        return Float.parseFloat(result);
    }
}
