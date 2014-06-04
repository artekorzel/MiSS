package pl.edu.agh.student.dpdsimulator;

public interface StartParameters {
    int numberOfSteps = 100;
    int numberOfDroplets = 10000;
    float deltaTime = 0.04f;
    float cutoffRadius = 0.01f;
    float boxSize = 1.0f;
    float velocityInitRange = 1.0f;
    float boltzmanConstant = 1 / 293.1f;/*(float) 1.3806488e-23;*/

}

//wiecej czastek, ~10mln - tutaj problemy z randomem i przeniesieniem na karte

//ewentualnie pozniej:
//przeniesc moze petle glowna tez na kernel
//wyciagnac wyznaczanie sasiadow przed obliczenia - przeniesienie do pamieci lokalnej i sprawdzenie wydajnosci


//wiecej typow czastek - rozbudowanie parametrow do macierzy (przynajmniej dwa typu czastek bo na sciany i erytrocyty)
//do plikow zrzucamy poza pozycja tez typ czastek
//jak budowac ksztalty (np sciane naczynia)?