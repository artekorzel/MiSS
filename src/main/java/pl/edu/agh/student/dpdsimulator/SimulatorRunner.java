package pl.edu.agh.student.dpdsimulator;

public class SimulatorRunner {

    public static void main(String[] args) {
        try {
            new TestSimulation().run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
