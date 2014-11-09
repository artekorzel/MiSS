package pl.edu.agh.student.dpdsimulator;

public class DpdSimulation {
    
    public static void main(String[] args) {
        try {
//            Simulation simulation = new GpuKernelSimulation();
            Simulation simulation = new JavaDpdMock();
            simulation.initData();
            simulation.performSimulation();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
