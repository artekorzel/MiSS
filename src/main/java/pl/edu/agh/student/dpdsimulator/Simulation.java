package pl.edu.agh.student.dpdsimulator;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Properties;
import pl.edu.agh.student.dpdsimulator.kernels.Dpd;

public abstract class Simulation {

    public static final int VECTOR_SIZE = 4;
    public static final double NANOS_IN_SECOND = 1000000000.0;
    public static final int numberOfCellNeighbours = 27;
    public static final int reductionLocalSize = 16;
    public static final int numberOfReductionGroups = 32;
    public static final int reductionSize = reductionLocalSize * numberOfReductionGroups;

    public static final String dataFileName = "simulation.data";
    public static final String csvHeader = "x,y,z,vx,vy,vz,t\n";
    public static final String psiHeaderBegining = "# PSI Format 1.0\n"
            + "#\n"
            + "# column[0] = \"x\"\n"
            + "# column[1] = \"y\"\n"
            + "# column[2] = \"z\"\n"
            + "# column[3] = \"vx\"\n"
            + "# column[4] = \"vy\"\n"
            + "# column[5] = \"vz\"\n"
            + "# column[6] = \"t\"\n"
            + "#\n"
            + "# symbol[3] = \"VX\"\n"
            + "# symbol[4] = \"VY\"\n"
            + "# symbol[5] = \"VZ\"\n"
            + "# symbol[6] = \"T\"\n"
            + "#\n"
            + "# type[3] = float\n"
            + "# type[4] = float\n"
            + "# type[5] = float \n"
            + "# type[6] = int\n"
            + "\n";
    public static String psiHeaderEnd = " 2694 115001\n"
            + "1.00 0.00 0.00\n"
            + "0.00 1.00 0.00\n"
            + "0.00 0.00 1.00\n\n";

    public static int numberOfSteps;
    public static double deltaTime;

    public static double boxSizeX;
    public static double boxSizeY;
    public static double boxSizeZ;
    public static double radiusIn;

    public static int cellsXAxis;
    public static int cellsYAxis;
    public static int cellsZAxis;
    public static double cellRadius;
    public static double averageDropletDistance;

    public static int numberOfCells;
    public static int numberOfDroplets;
    public static int numberOfRandoms;
    public static int maxDropletsPerCell;
    public static int numberOfCellKinds;

    public static boolean shouldStoreCSVFiles;
    public static boolean shouldStorePSIFiles;
    public static boolean shouldPrintAvgVelocity;
    public static boolean shouldPrintKineticEnergy;
    public static boolean shouldPrintVelocityProfile;
    public static boolean shouldSimulateVesselDroplets;
    public static String resultsDirectoryBase;
    public static int stepDumpThreshold;
    public static double accelerationVesselPart;
    public static double accelerationValue;
    public static int accelerationVeselSteps;
    public static double initialVelocity;
    public static double randomForceMultiplier;
    
    public static boolean generateRandomPositions;

    public static double[] mass;
    public static double[] avgTempVelocity;
    public static double[][] cutOffRadius;
    public static double[][] pi;
    public static double[][] gamma;
    public static double[][] sigma;
    
    public static double fe;
    public static double ft;
    public static double Rhod = 8.44e+26;
    public static double Boltz = 1.3806e-23;
    public static double tempd = 306.0;    

    public abstract void initData() throws Exception;

    public abstract void performSimulation() throws Exception;

    protected Dpd.DropletParameters createDropletParameter(double mass) {
        Dpd.DropletParameters dropletParameter = new Dpd.DropletParameters();
        dropletParameter.mass(mass);
        return dropletParameter;
    }

    protected Dpd.PairParameters createPairParameter(double cutoffRadius, double pi, double gamma, double sigma) {
        Dpd.PairParameters pairParameter = new Dpd.PairParameters();
        pairParameter.cutoffRadius(cutoffRadius);
        pairParameter.pi(pi);
        pairParameter.sigma(sigma);
        pairParameter.gamma(gamma);
        return pairParameter;
    }

    protected void loadInitialDataFromFile(String fileName) {
        try {
            Properties prop = getProperties(fileName);

            shouldStoreCSVFiles = Boolean.parseBoolean(prop.getProperty("shouldStoreCSVFiles"));
            shouldStorePSIFiles = Boolean.parseBoolean(prop.getProperty("shouldStorePSIFiles"));
            shouldPrintAvgVelocity = Boolean.parseBoolean(prop.getProperty("shouldPrintAvgVelocity"));
            shouldPrintKineticEnergy = Boolean.parseBoolean(prop.getProperty("shouldPrintKineticEnergy"));
            shouldPrintVelocityProfile = Boolean.parseBoolean(prop.getProperty("shouldPrintVelocityProfile"));
            numberOfSteps = Integer.parseInt(prop.getProperty("numberOfSteps"));
            deltaTime = Double.parseDouble(prop.getProperty("deltaTime"));
            cellsXAxis = Integer.parseInt(prop.getProperty("cellsXAxis"));
            cellsYAxis = Integer.parseInt(prop.getProperty("cellsYAxis"));
            cellsZAxis = Integer.parseInt(prop.getProperty("cellsZAxis"));
            averageDropletDistance = Double.parseDouble(prop.getProperty("averageDropletDistance"));
            maxDropletsPerCell = Integer.parseInt(prop.getProperty("maxDropletsPerCell"));
            numberOfDroplets = Integer.parseInt(prop.getProperty("numberOfDroplets"));
            numberOfCellKinds = Integer.parseInt(prop.getProperty("numberOfCellKinds"));
            generateRandomPositions = Boolean.parseBoolean(prop.getProperty("generateRandomPositions"));
            radiusIn = Double.parseDouble(prop.getProperty("radiusIn"));
            resultsDirectoryBase = prop.getProperty("resultsDirectoryBase");
            stepDumpThreshold = Integer.parseInt(prop.getProperty("stepDumpThreshold"));
            accelerationVesselPart = Double.parseDouble(prop.getProperty("accelerationVesselPart"));
            accelerationValue = Double.parseDouble(prop.getProperty("accelerationValue"));
            accelerationVeselSteps = Integer.parseInt(prop.getProperty("accelerationVeselSteps"));
            shouldSimulateVesselDroplets = Boolean.parseBoolean(prop.getProperty("shouldSimulateVesselDroplets"));
            initialVelocity = Double.parseDouble(prop.getProperty("initialVelocity"));
            randomForceMultiplier = Double.parseDouble(prop.getProperty("randomForceMultiplier"));
            Rhod = Double.parseDouble(prop.getProperty("rhod"));
            Boltz = Double.parseDouble(prop.getProperty("boltz")); 
            tempd = Double.parseDouble(prop.getProperty("tempd"));
            deltaTime = Double.parseDouble(prop.getProperty("deltaTime"));
            
            numberOfRandoms = numberOfDroplets * (numberOfDroplets - 1) / 2;

            mass = new double[numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                mass[i] = Double.parseDouble(prop.getProperty("mass(" + i + ")"));
            }

            cutOffRadius = new double[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    cutOffRadius[i][j] = cutOffRadius[j][i] = Double.parseDouble(prop.getProperty("cutoffRadius(" + i + "," + j + ")"));
                }
            }

            pi = new double[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    pi[i][j] = pi[j][i] = Double.parseDouble(prop.getProperty("pi(" + i + "," + j + ")"));
                }
            }

            gamma = new double[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    gamma[i][j] = gamma[j][i] = Double.parseDouble(prop.getProperty("gamma(" + i + "," + j + ")"));
                }
            }

            sigma = new double[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    sigma[i][j] = sigma[j][i] = Double.parseDouble(prop.getProperty("sigma(" + i + "," + j + ")"));
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        avgTempVelocity = new double[numberOfCellKinds];
        for (int i = 0; i < numberOfCellKinds; i++) {
            avgTempVelocity[i] = 0.0;
        }
        
        cellRadius = getGreatestCutOffRadius();
        boxSizeX = cellsXAxis / 2;
        boxSizeY = cellsYAxis / 2;
        boxSizeZ = cellsZAxis / 2;
        numberOfCells = cellsXAxis * cellsYAxis * cellsZAxis;
        System.out.println("" + boxSizeX + ", " + boxSizeY + ", " + boxSizeZ + "; " + numberOfDroplets + "; " + numberOfCells);
                
        double ul = Math.cbrt(numberOfDroplets / (Rhod * numberOfCells));
        double ue = mass[0] * ul / deltaTime * ul / deltaTime;
        fe = ue / numberOfDroplets;
        ft = 1.0 / (1.5 * Boltz);
    }

    private Properties getProperties(String fileName) throws Exception {
        Properties prop = new Properties();
        InputStream inputStream = new FileInputStream(fileName);
        prop.load(inputStream);
        return prop;
    }
    
    private double getGreatestCutOffRadius() {
        double max = 0.0;
        for(int i = 0; i < numberOfCellKinds; i++){
            max = max > cutOffRadius[0][i] ? max : cutOffRadius[0][i];
        }
        return max;
    }
}
