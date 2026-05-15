# Obserwacje z badań
Obserwacje z badań opierają się na metodycznym podziale testów na cztery główne warianty:
1) LSTM bez Early Stopping
<img width="945" height="451" alt="image" src="https://github.com/user-attachments/assets/4f904879-12b3-454a-96ce-11655d12e6a0" />
<img width="945" height="477" alt="image" src="https://github.com/user-attachments/assets/c8ad0b11-d49d-4ed8-9840-7532ece77c3f" />
<img width="945" height="483" alt="image" src="https://github.com/user-attachments/assets/cf2f5147-6dcc-42e9-9fda-ad39852384e5" />
<img width="945" height="485" alt="image" src="https://github.com/user-attachments/assets/976c4b6b-bb0b-43c0-b451-eef696d43bf0" />
<img width="945" height="479" alt="image" src="https://github.com/user-attachments/assets/088b07ed-d871-4637-adad-687200abdfb3" />
<img width="945" height="366" alt="image" src="https://github.com/user-attachments/assets/8cc9f78b-f92d-42b9-99a1-091117c9b17a" />
<img width="925" height="288" alt="image" src="https://github.com/user-attachments/assets/7f7e8b15-3daa-46cd-a96c-83b779292a68" />
<img width="672" height="286" alt="image" src="https://github.com/user-attachments/assets/395b60af-0762-48bd-b324-82c136689dc1" />
   
2) LSTM z Early Stopping
<img width="945" height="471" alt="image" src="https://github.com/user-attachments/assets/56588e64-9a95-4cb7-bef7-1aa3e4bb69bf" />
<img width="945" height="460" alt="image" src="https://github.com/user-attachments/assets/13bd03f3-ed4d-4b2f-bb16-3ed67fdd398e" />
<img width="945" height="283" alt="image" src="https://github.com/user-attachments/assets/386f5776-7942-4990-b4f5-6a4f74e4ca58" />
<img width="667" height="308" alt="image" src="https://github.com/user-attachments/assets/6f9c249f-0643-4861-84cb-7a6150bb207b" />

3) GRU bez Early Stopping
<img width="945" height="453" alt="image" src="https://github.com/user-attachments/assets/4afec97d-b160-45f3-9693-53863a91c665" />
<img width="945" height="486" alt="image" src="https://github.com/user-attachments/assets/32a35e9f-be69-422e-8589-b1d45e657348" />
<img width="945" height="473" alt="image" src="https://github.com/user-attachments/assets/f0b487ca-06d2-4a6f-aae9-36092808e33d" />
<img width="945" height="464" alt="image" src="https://github.com/user-attachments/assets/c5ecd018-8aa4-4bc9-9059-1849dc567d1a" />
<img width="945" height="477" alt="image" src="https://github.com/user-attachments/assets/fbd7a9ca-5e14-46ff-afd6-ec73912e21e8" />
<img width="874" height="403" alt="image" src="https://github.com/user-attachments/assets/d2e7524f-e70b-478c-9593-a9f2b41fe708" />
<img width="945" height="283" alt="image" src="https://github.com/user-attachments/assets/dd161764-eb9a-4e9e-82dd-19b8218f1821" />
<img width="697" height="291" alt="image" src="https://github.com/user-attachments/assets/2353c3dd-beef-48ef-8fa0-8b6aafcc9ad8" />

5) GRU z Early Stopping
<img width="945" height="452" alt="image" src="https://github.com/user-attachments/assets/62e08d01-43e6-4312-a1b5-2aa88e1b9dcb" />
<img width="945" height="829" alt="image" src="https://github.com/user-attachments/assets/7b89b7f9-5888-43ee-85b5-ac20ae88f485" />
<img width="945" height="384" alt="image" src="https://github.com/user-attachments/assets/9493abde-693f-47d2-b055-0b48ce5aa405" />
<img width="945" height="267" alt="image" src="https://github.com/user-attachments/assets/b80b1e95-7c48-4a8c-8166-23e6d21759dc" />
<img width="731" height="275" alt="image" src="https://github.com/user-attachments/assets/c6244705-b1b8-4e0f-aaa5-b76bd263fe2a" />

Taka struktura wskazuje na bezpośrednie porównanie zachowania modeli uczonych „na sztywno” (przez określoną z góry liczbę epok, np. 100) z modelami wyposażonymi w mechanizm obronny przed przeuczeniem. Eksperyment potwierdza praktyczne różnice — zastosowanie walidacji i wczesnego zatrzymywania („Early Stopping”) odcina proces treningu w punkcie, w którym błąd na zbiorze walidacyjnym (val_loss) przestaje maleć, co skutecznie chroni przed przetrenowaniem (overfittingiem) i często redukuje fizyczny czas całego procesu uczenia.
# Najlepsza konfiguracja dla predykcji
Na podstawie obliczeń i uzyskanych współczynników błędów, najlepszą kombinacją dla tego konkretnego przypadku okazała się poniższa:

Optymalizator: Adam

Funkcja straty: Huber

Architektura: LSTM

Ilość units: 100

Uzasadnienie:
Taka konfiguracja osiągnęła ostateczny błąd średniokwadratowy (RMSE) równy 1.66, z powodzeniem spełniając założenie zadania (RMSE < 2.0). Konkurencyjny model GRU z dokładnie tymi samymi parametrami osiągnął wynik gorszy (RMSE = 1.80).
Ilość jednostek (100 units): Zwiększona liczba jednostek ukrytych zapewniła sieci odpowiednią "pojemność" pamięci, umożliwiając wyłapanie subtelniejszych długoterminowych zależności w cenach akcji IBM (kolumna Close).
Optymalizator (Adam): Pozwolił na niezwykle płynne adaptowanie szybkości uczenia do danych, bez ryzyka utknięcia w suboptymalnym minimum lokalnym.
Funkcja straty (Huber): Działa jako świetny kompromis pomiędzy błędem MSE i MAE. Funkcja Hubera jest znacznie mniej wrażliwa na gwałtowne odchylenia (outliery), które są naturalnym zjawiskiem na rynku giełdowym, dzięki czemu uczenie odbywało się stabilniej.
Powstrzymanie przetrenowania: Dołączenie do treningu (zgodnie z przygotowanymi wywołaniami callbacks) mechanizmu EarlyStopping zapobiega "pamięciowemu" przyswajaniu danych przez model. Dzięki walidacji (validation_split=0.1) wagi powracają do swojej najlepszej formy.

# Prędkość działania: GRU a LSTM
Z logów środowiska wykonawczego jednoznacznie wynika, że architektura GRU działała zauważalnie szybciej na poziomie przetwarzania poszczególnych partii danych (batchy).

Średni czas wykonania pojedynczego kroku (step) dla GRU wynosił około 14–15 ms.

Średni czas wykonania kroku dla LSTM wynosił zazwyczaj 15–20 ms (z częstymi skokami do 22-23 ms).

Wynika to ze specyfiki obu komórek. Sieć GRU jest prostszą architekturą – nie posiada pamięci wewnętrznej komórki (cell state) i składa się tylko z dwóch bramek (aktualizacji i resetu), podczas gdy LSTM operuje na trzech (zapominania, wejścia i wyjścia). Dzięki temu uczy się nieco szybciej z racji mniejszej liczby operacji matematycznych, co było wyraźnie widać w czasach logowanych dla poszczególnych epok. Mimo to, w tym zadaniu, to nieznacznie wolniejsze LSTM zapewniło lepszą jakość predykcji.
