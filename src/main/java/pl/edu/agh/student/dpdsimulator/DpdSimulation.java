package pl.edu.agh.student.dpdsimulator;

public class DpdSimulation {
    
    static float boxSizeScale = 1.0f;
    static float boxWidthScale = 1.0f;
    static int numberOfDroplets = 62500;
    
    public static void main(String[] args) {
        try {
            Simulation simulation = new GpuKernelSimulation();
//            Simulation simulation = new JavaDpdMock();
            simulation.initData(boxSizeScale, boxWidthScale, numberOfDroplets);
            simulation.performSimulation();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
