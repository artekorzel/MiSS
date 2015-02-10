package pl.edu.agh.student.dpdsimulator;

public class DpdSimulation {
    
    static float boxSizeScale = 1f;
    static float boxWidthScale = 1f;
    static int numberOfDroplets = Simulation.baseNumberOfDroplets / 64;
    
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
