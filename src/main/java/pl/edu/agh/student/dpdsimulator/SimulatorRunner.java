package pl.edu.agh.student.dpdsimulator;

import java.io.IOException;

public class SimulatorRunner {

    public static void main(String[] args) {
        try {
            new TestSimulation().run();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
