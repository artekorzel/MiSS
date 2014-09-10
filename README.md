-- Uruchomienie --

Projekt wykorzystuje mavena do zarządzania zależnościami. Jest on odpowiednio skonfigurowany, aby można całość uruchomić za pomocą polecenia: "mvn exec:java". 
Można również wydać polecenie: "mvn package" w celu utworzenia pliku JAR wraz z zależnościami.

Konfiguracja wartości parametrów następuje obecnie poprzez edycję klasy pl.edu.agh.student.dpdsimulator.DpdSimulation.

Program generuje pliki wyjściowe w katalogu "../results_<timestamp>/" względem katalogu głównego projektu. Pliki te posiadają nazwy result<numer_kroku>.csv, w kolejnych liniach znajdują się współrzędne położenia po danym kroku obliczeń oraz typ cząstki.