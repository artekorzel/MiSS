/*
Struktura przetrzymujaca parametry specyficzne dla typu czastki
*/
typedef struct DropletParameter {
    float cutoffRadius;//promien odciecia
    float mass;//masa czastki
    float repulsionParameter;//wspolczynnik odpychania
    float lambda;
    float sigma;
    float gamma;
} DropletParameter;

/*
Funkcja przyjmuje odleglosc pomiedzy czastkami oraz maksymalna odleglosc dla ktorej powinna 
wykonywac obliczenia i zwraca wartosc wagi dla sily konserwatywnej.
*/
float weightR(float distanceValue, float cutoffRadius) {
    if(distanceValue > cutoffRadius) {
        return 0.0;
    }
    return 1.0 - distanceValue / cutoffRadius;
}

/*
Funkcja przyjmuje wektor pozycji i wielkosc pudla obliczeniowego 
i zwraca pozycje czastki znormalizowana do wielkosci pudla.
*/
float3 normalizePosition(float3 vector, float boxSize) {
    return fmod(fmod(vector + boxSize, 2.0f * boxSize) + 2.0f * boxSize, 2.0f * boxSize) - boxSize;
}

/*
Funkcja hashujaca dla pary czastek na podstawie ich ID. Dzieki niej jestesmy w stanie 
dla danej pary wygenerowac taka sama wartosc wspolczynnika sily brownowskiej przy obliczeniach 
z punktu widzenia obu czastek z pary, nie musimy przechowywac tablicy wspolczynnikow.
*/
int calculateHash(int d1, int d2) {    
    int i1, i2;
    if(d1 <= d2) {
        i1 = d1;
        i2 = d2;
    } else {
        i1 = d2;
        i2 = d1;
    }
    return (i1 + i2) * (i1 + i2 + 1) / 2 + i2;
}

/*
Funkcja randomizujaca z rozkladem liniowym na przedziale <-1; 1>
*/
float rand(int* seed, int step) {
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = (*seed * a * step) % m;
    return (float)(*seed) / (m - 1);
}

/*
Funkcja randomizujaca z rozkladem normalnym na przedziale <-1; 1>
*/
float normalRand(float U1, float U2) {
     float R = -2 * log(U1);
     float fi = 2 * M_PI * U2;
     float Z1 = sqrt(R) * cos(fi);
     return Z1;
     //float Z2 = sqrt(R) * sin(fi);
}

/*
Funkcja randomizujaca z rozkladem Gaussa na przedziale <-1; 1>
*/
float gaussianRandom(int dropletId, int neighbourId, int numberOfDroplets, int step) {
    int seed = calculateHash(dropletId, neighbourId);
    float U1 = (rand(&seed, step) + 1.0) / 2;
    float U2 = (rand(&seed, step) + 1.0) / 2;
    return normalRand(U1, U2);
}

/*
Funkcja obliczajaca sile dzialajaca na dana czastke jako sume 3 skladowych sily wyjsciowej: 
konserwatywnej, dyssypatywnej oraz brownowskiej dla kazdej czastki w odleglosci mniejszej, 
niz promien odciecia (wyjatek stanowi oddzialywanie par czastek sciany, 
dla ktorych zastosowano prostszy algorytm symulujacy przyciaganie czastek).
*/
float3 calculateForce(global float3* positions, global float3* velocities, global DropletParameter* params,
        global int* types, int numberOfDroplets, int dropletId, int step) {

    float3 conservativeForce = (float3)(0.0, 0.0, 0.0);
    float3 dissipativeForce = (float3)(0.0, 0.0, 0.0);
    float3 randomForce = (float3)(0.0, 0.0, 0.0);

    float3 dropletPosition = positions[dropletId];
    float3 dropletVelocity = velocities[dropletId];
    
    int dropletType = types[dropletId];
    DropletParameter dropletParameter = params[dropletType];
    
    float repulsionParameter = dropletParameter.repulsionParameter;
    float gamma = dropletParameter.gamma;
    float sigma = dropletParameter.sigma;
    
    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float3 neighbourPosition = positions[neighbourId];
            float distanceValue = distance(neighbourPosition, dropletPosition);
            
            int neighbourType = types[neighbourId];
            DropletParameter neighbourParameter = params[neighbourType];
            float cutoffRadius = neighbourParameter.cutoffRadius;
            
            if(distanceValue < cutoffRadius) {
                float3 normalizedPositionVector = normalize(neighbourPosition - dropletPosition);
                if(dropletType != 0 || neighbourType != 0) {
                    float weightRValue = weightR(distanceValue, cutoffRadius);
                    float weightDValue = weightRValue * weightRValue;

                    conservativeForce -= sqrt(repulsionParameter * neighbourParameter.repulsionParameter)
                            * (1.0f - distanceValue / cutoffRadius) * normalizedPositionVector;

                    dissipativeForce += gamma * weightDValue * normalizedPositionVector
                            * dot(normalizedPositionVector, velocities[neighbourId] - dropletVelocity);

                    randomForce -= sigma * weightRValue * normalizedPositionVector
                            * gaussianRandom(dropletId, neighbourId, numberOfDroplets, step);
                }
            }
        }
    }

    if(dropletType == 0 || step > 100) {
        return conservativeForce + dissipativeForce + randomForce;
    } else {
        return conservativeForce + dissipativeForce + randomForce + (float3)(0.0f, 0.01f, 0.0f);    
    }
}

/*
Metoda kernela uruchamiana w kazdym kroku symulacji w celu wyznaczenia sil dla kazdej czastki.
*/
kernel void calculateForces(global float3* positions, global float3* velocities, global float3* forces, 
        global DropletParameter* params, global int* types, int numberOfDroplets, int step) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    forces[dropletId] = calculateForce(positions, velocities, params, types,
             numberOfDroplets, dropletId, step);
}

/*
Funkcja generujaca pozycje oraz przewidywane predkosci, ktore potem sa weryfikowane. 
Przewidywane predkosci wykorzystywane sa przy obliczaniu sil czastek w danym kroku.
*/
kernel void calculateNewPositionsAndPredictedVelocities(global float3* positions, global float3* velocities,
        global float3* forces, global float3* newPositions, global float3* predictedVelocities,
        global DropletParameter* params, global int* types, float deltaTime, int numberOfDroplets, float boxSize) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    DropletParameter dropletParameter = params[types[dropletId]];
    float3 dropletVelocity = velocities[dropletId];
    float3 dropletForce = forces[dropletId];
    float dropletMass = dropletParameter.mass;

    float3 newPosition = positions[dropletId] + deltaTime * dropletVelocity
            + 0.5f * deltaTime * deltaTime * dropletForce / dropletMass;
            
    newPositions[dropletId] = normalizePosition(newPosition, boxSize);
    
    predictedVelocities[dropletId] = dropletVelocity 
            + dropletParameter.lambda * deltaTime * dropletForce / dropletMass;
}

/*
W tej funkcji kazdym kroku obliczane sa przewidywane wartosci sily, a nastepnie 
na ich podstawie obliczane sa predkosci rzeczywiste jakie czastki osiagaja po danym kroku.
*/
kernel void calculateNewVelocities(global float3* newPositions, global float3* velocities,
        global float3* predictedVelocities, global float3* newVelocities, global float3* forces,
        global DropletParameter* params, global int* types, float deltaTime, int numberOfDroplets, int step) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float3 predictedForce = calculateForce(newPositions, predictedVelocities, params, 
            types, numberOfDroplets, dropletId, step);

    newVelocities[dropletId] = velocities[dropletId] + 0.5f * deltaTime 
            * (forces[dropletId] + predictedForce) / params[types[dropletId]].mass;
}

/*
Funkcja obliczajaca rownolegle srednia predkosc wszystkich czastek.
*/
kernel void doVectorReduction(global float3* data, global float3* partialSums, 
        global float3* output, int dataLength) {

    int global_id = get_global_id(0);
    int group_size = get_global_size(0);

    if(global_id < dataLength) {
        partialSums[global_id] = data[global_id];
    } else {
        partialSums[global_id] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    int offset;
    for(offset = group_size/2; offset > 0; offset >>= 1) {
        if(global_id < offset){
            partialSums[global_id] += partialSums[global_id + offset];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    
    if(global_id == 0) {
        output[0] = partialSums[0]/dataLength;
    }
}
