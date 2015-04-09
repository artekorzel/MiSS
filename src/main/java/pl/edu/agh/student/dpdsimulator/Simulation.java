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
    public static float deltaTime;

    public static float boxSize;
    public static float boxWidth;
    public static float radiusIn;

    public static int cellsXAxis;
    public static int cellsYAxis;
    public static int cellsZAxis;
    public static float cellRadius;
    public static float averageDropletDistance;

    public static int numberOfCells;
    public static int numberOfDroplets;
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
    public static float accelerationVesselPart;
    public static float accelerationValue;
    public static int accelerationVeselSteps;

    public static boolean generateRandomPositions;

    public static float[] mass;
    public static float[][] cutOffRadius;
    public static float[][] pi;
    public static float[][] gamma;
    public static float[][] sigma;

    public abstract void initData() throws Exception;

    public abstract void performSimulation() throws Exception;

    protected Dpd.DropletParameters createDropletParameter(float mass) {
        Dpd.DropletParameters dropletParameter = new Dpd.DropletParameters();
        dropletParameter.mass(mass);
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

    protected void loadInitialDataFromFile(String fileName) {
        try {
            Properties prop = getProperties(fileName);

            shouldStoreCSVFiles = Boolean.parseBoolean(prop.getProperty("shouldStoreCSVFiles"));
            shouldStorePSIFiles = Boolean.parseBoolean(prop.getProperty("shouldStorePSIFiles"));
            shouldPrintAvgVelocity = Boolean.parseBoolean(prop.getProperty("shouldPrintAvgVelocity"));
            shouldPrintKineticEnergy = Boolean.parseBoolean(prop.getProperty("shouldPrintKineticEnergy"));
            shouldPrintVelocityProfile = Boolean.parseBoolean(prop.getProperty("shouldPrintVelocityProfile"));
            numberOfSteps = Integer.parseInt(prop.getProperty("numberOfSteps"));
            deltaTime = Float.parseFloat(prop.getProperty("deltaTime"));
            cellsXAxis = Integer.parseInt(prop.getProperty("cellsXAxis"));
            cellsYAxis = Integer.parseInt(prop.getProperty("cellsYAxis"));
            cellsZAxis = Integer.parseInt(prop.getProperty("cellsZAxis"));
            averageDropletDistance = Float.parseFloat(prop.getProperty("averageDropletDistance"));
            maxDropletsPerCell = Integer.parseInt(prop.getProperty("maxDropletsPerCell"));
            numberOfDroplets = Integer.parseInt(prop.getProperty("numberOfDroplets"));
            numberOfCellKinds = Integer.parseInt(prop.getProperty("numberOfCellKinds"));
            generateRandomPositions = Boolean.parseBoolean(prop.getProperty("generateRandomPositions"));
            radiusIn = Float.parseFloat(prop.getProperty("radiusIn"));
            resultsDirectoryBase = prop.getProperty("resultsDirectoryBase");
            stepDumpThreshold = Integer.parseInt(prop.getProperty("stepDumpThreshold"));
            accelerationVesselPart = Float.parseFloat(prop.getProperty("accelerationVesselPart"));
            accelerationValue = Float.parseFloat(prop.getProperty("accelerationValue"));
            accelerationVeselSteps = Integer.parseInt(prop.getProperty("accelerationVeselSteps"));
            shouldSimulateVesselDroplets = Boolean.parseBoolean(prop.getProperty("shouldSimulateVesselDroplets"));

            mass = new float[numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                mass[i] = Float.parseFloat(prop.getProperty("mass(" + i + ")"));
            }

            cutOffRadius = new float[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    cutOffRadius[i][j] = cutOffRadius[j][i] = Float.parseFloat(prop.getProperty("cutoffRadius(" + i + "," + j + ")"));
                }
            }

            pi = new float[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    pi[i][j] = pi[j][i] = Float.parseFloat(prop.getProperty("pi(" + i + "," + j + ")"));
                }
            }

            gamma = new float[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    gamma[i][j] = gamma[j][i] = Float.parseFloat(prop.getProperty("gamma(" + i + "," + j + ")"));
                }
            }

            sigma = new float[numberOfCellKinds][numberOfCellKinds];
            for (int i = 0; i < numberOfCellKinds; i++) {
                for (int j = i; j < numberOfCellKinds; j++) {
                    sigma[i][j] = sigma[j][i] = Float.parseFloat(prop.getProperty("sigma(" + i + "," + j + ")"));
                }
            }
            
            cellRadius = getGratestCutOffRadius();
            boxSize = cellRadius * cellsXAxis / 2;
            boxWidth = cellRadius * cellsYAxis / 2;
            numberOfCells = cellsXAxis * cellsYAxis * cellsZAxis;

            System.out.println("" + boxSize + ", " + boxWidth + "; " + numberOfDroplets + "; " + numberOfCells);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Properties getProperties(String fileName) throws Exception {
        Properties prop = new Properties();
        InputStream inputStream = new FileInputStream(fileName);
        prop.load(inputStream);
        return prop;
    }

    private float getGratestCutOffRadius() {
        float max = 0.0f;
        for(int i = 0; i < numberOfCellKinds; i++){
            max = max > cutOffRadius[0][i] ? max : cutOffRadius[0][i];
        }
        return max;
    }
}