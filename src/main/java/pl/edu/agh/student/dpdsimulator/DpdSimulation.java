package pl.edu.agh.student.dpdsimulator;

public class DpdSimulation {    
    
    
    public static void main(String[] args) {
        try {
            Simulation simulation = new GpuKernelSimulation();
            simulation.loadInitialDataFromFile(Simulation.dataFileName);
            simulation.scaleParameters();
            simulation.initData();
            simulation.performSimulation();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
