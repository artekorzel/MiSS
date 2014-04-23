package pl.edu.agh.student.dpdsimulator;

public interface StartParameters {
    int numberOfSteps = 10;
    int numberOfDroplets = 100;
    float deltaTime = 0.04f;
    float cutoffRadius = 0.01f;
    float boxSize = 1.0f;
    float velocityInitRange = 1.0f;
    float temperature = 293.1f;
    float boltzmanConstant = 1 / 293.1f;/*(float) 1.3806488e-23;*/
    float density = 3.0f;
    float repulsionParameter = 75.0f * boltzmanConstant * temperature / density;
    float lambda = 0.5f;
    float sigma = 0.075f;
    float gamma = sigma * sigma / 2.0f / boltzmanConstant / temperature;
}

//wyciagnac wyznaczanie sasiadow przed obliczenia??? - ale jak?
//dodac srednia predkosc czastek po kazdej iteracji - w jaki sposob?
//przeniesc moze petle glowna tez na kernel