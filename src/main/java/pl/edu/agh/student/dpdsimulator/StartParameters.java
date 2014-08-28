package pl.edu.agh.student.dpdsimulator;

public interface StartParameters {

    int numberOfSteps = 100;
    int numberOfDroplets = 30000;
    float deltaTime = 1.0f;
    float cutoffRadius = 0.1f;
    float boxSize = 26.0f;
    float temperature = 310.0f;
    float boltzmanConstant = 1 / temperature;

}

//algorytm podzialu na celki