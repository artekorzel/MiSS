package pl.edu.agh.student.dpdsimulator;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
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

    public static float boxSizeX;
    public static float boxSizeY;
    public static float boxSizeZ;
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
    
    public static double fe;
    public static double ft;
    public static double Rhod = 8.44e+26;
    public static double Boltz = 1.3806e-23;
    public static double tempd = 306.0;

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

    protected void scaleParameters() {        
        int i, j, ld, ld1;
        double coef1, coef2, smass, rc, sep;
        
        ld = 3;
        ld1 = 3;
        coef1 = 2.0 / 7.0;
        coef2 = 3.0 / 5.0;
        for (i = 0; i < numberOfCellKinds; i++) {
            if (mass[i] > 1.0e-20) {
                mass[i] /= 1000.0f * 6.0230e+23f;
            }
        }
        
        for (i = 0; i < numberOfCellKinds; i++) {
            for (j = 0; j < numberOfCellKinds; j++) {
                smass = 2.0 * mass[i] / (mass[i] + mass[j]) * mass[j];
                rc = cutOffRadius[i][j] / Math.cbrt(Rhod);
                cutOffRadius[i][j] = (float) rc;
                gamma[i][j] = (float) (gamma[i][j] * (ld1 * Math.sqrt(Boltz * tempd / smass)) / rc);
                gamma[i][j] = 10.0f * gamma[i][j];
                pi[i][j] = (float) (6.0 * 2.0 * pi[i][j] * ld / (Rhod * coef2 * rc));
            }
        }
        
        cellRadius = getGreatestCutOffRadius();
        
        sep = 4.0 * Math.PI * cellRadius * cellRadius * Rhod * cellRadius / 3.0;
        for (i = 0; i < numberOfCellKinds; i++) {
            for (j = 0; j < numberOfCellKinds; j++) {
                smass = 2.0 * mass[i] / (mass[i] + mass[j]) * mass[j];
                sigma[i][j] = (float) (Math.sqrt(2.0 * Boltz * tempd) * Math.sqrt(gamma[i][j] * smass * sep));
            }
        }
        
        double ul = Math.cbrt(numberOfDroplets / (Rhod * cellsXAxis * cellsYAxis * cellsZAxis));
        System.out.println(String.format("Scalep rcmax %e ul %e\n", cellRadius, ul));
        System.out.println(String.format("Scalep nfx %d nfy %d nfz %d\n", cellsXAxis, cellsYAxis, cellsZAxis));
        cellsXAxis = (int) (ul * cellsXAxis / cellRadius);
        cellsYAxis = (int) (ul * cellsYAxis / cellRadius);
        cellsZAxis = (int) (ul * cellsZAxis / cellRadius);
        System.out.println(String.format("Scalep ncx %d ncy %d ncz %d\n", cellsXAxis, cellsYAxis, cellsZAxis));
        
        ul = Math.cbrt(numberOfDroplets / (Rhod * cellsXAxis * cellsYAxis * cellsZAxis));
        double ue = mass[0] * ul / deltaTime * ul / deltaTime;
        System.out.println(String.format("Scalep ue: %e %e %e %e\n", mass[0], ue, ul, deltaTime));
        
        fe = ue / numberOfDroplets;
        ft = 1.0 / (1.5 * Boltz);
        System.out.println(String.format("Scalep fe: %e ft: %e\n", fe, ft));
        
        for (i = 0; i < numberOfCellKinds; i++) {
            for (j = 0; j < numberOfCellKinds; j++) {
                pi[i][j] = (float) (pi[i][j] * deltaTime * deltaTime / (ul * mass[0]));
                gamma[i][j] = gamma[i][j] * deltaTime;
                sigma[i][j] = (float) ((sigma[i][j] * Math.sqrt(3.0) * deltaTime * Math.sqrt(deltaTime)) / (ul * mass[0]));
                System.out.println(String.format("SH %g %g %g\n", pi[i][j], gamma[i][j], sigma[i][j]));
            }
        }
        
        for (i = 1; i < numberOfCellKinds; i++) {
            mass[i] = mass[i] / mass[0];
        }
        mass[0] = 1.0f;
        
        for (i = 0; i < numberOfCellKinds; i++) {
            for (j = 0; j < numberOfCellKinds; j++) {
                cutOffRadius[i][j] = (float) (cutOffRadius[i][j] / ul);
                System.out.println(String.format("rcut : %e\n", cutOffRadius[i][j]));
            }
        }
        
        cellRadius = getGreatestCutOffRadius();
        boxSizeX = cellsXAxis / 2;
        boxSizeY = cellsYAxis / 2;
        boxSizeZ = cellsZAxis / 2;
        numberOfCells = cellsXAxis * cellsYAxis * cellsZAxis;
        deltaTime = 1f;
        System.out.println("" + boxSizeX + ", " + boxSizeY + ", " + boxSizeZ + "; " + numberOfDroplets + "; " + numberOfCells);
        
        System.out.println("Pi " + Arrays.toString(pi[0]));
        System.out.println("Gamma " + Arrays.toString(gamma[0]));
        System.out.println("Sigma " + Arrays.toString(sigma[0]));
    }
    
    private float getGreatestCutOffRadius() {
        float max = 0.0f;
        for(int i = 0; i < numberOfCellKinds; i++){
            max = max > cutOffRadius[0][i] ? max : cutOffRadius[0][i];
        }
        return max;
    }
}
