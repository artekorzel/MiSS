package pl.edu.agh.student.dpdsimulator;

public class SimulatorRunner {

    public static void main(String[] args) {
        try {
            new DpdSimulation().run();
//            new DpdMockSimulation().run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
