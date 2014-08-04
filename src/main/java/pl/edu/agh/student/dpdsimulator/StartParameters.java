package pl.edu.agh.student.dpdsimulator;

public interface StartParameters {

    int numberOfSteps = 10;
    int numberOfDroplets = 10000;
    float deltaTime = 1.0f;
    float cutoffRadius = 0.99f;
    float boxSize = 26.0f;
    float velocityInitRange = 0.05f;
    float boltzmanConstant = 1 / 310.0f;/*(float) 1.3806488e-23;*/

}

//wiecej czastek, ~10mln - tutaj niezidentyfikowany problem z pamiecia, mozliwe ok 1mln
//przeniesc moze petle glowna tez na kernel - nie bedzie sie dalo (?) ze wzgledu na koniecznosc zapisywania pozycji itp
//wyciagnac wyznaczanie sasiadow przed obliczenia - przeniesienie do pamieci lokalnej i sprawdzenie wydajnosci - nie poprawia wydajnosci

//wiecej typow czastek - rozbudowanie parametrow do macierzy (przynajmniej dwa typu czastek bo na sciany i erytrocyty)
//do plikow zrzucamy poza pozycja tez typ czastek
//jak budowac ksztalty (np sciane naczynia)?

//parametry z pracy habilitacyjnej prof Boryczki
//tam też algo podziału na celki


//trzy typy czastek a nie dwa - krwinki, osocze i naczynia - FB

//naczynie - czastki tylko dookola krwinek a nie w calej przestrzeni (dwa promienie?) - FB

//dodac masy, defaultowo masa=1 dla krwinek - AO

//uruchomienie na zeusie - czy sie da, sprobowac - AO

//algorytm podzialu na celki