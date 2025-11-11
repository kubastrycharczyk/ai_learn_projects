### 1. Tytuł Projektu

Implementacja, Porównanie i Stabilizacja Głębokich Sieci Neuronowych w PyTorch z Wykorzystaniem Zasad OOP.

### 2. Cel Projektu

Celem projektu jest zaprojektowanie, implementacja i wytrenowanie modelu głębokiego uczenia (MLP - Multi-Layer Perceptron) do zadania klasyfikacji binarnej. Kluczowe jest zastosowanie programowania obiektowego (OOP) do definicji zbioru danych i modelu oraz systematyczne zbadanie wpływu różnych optymalizatorów, funkcji aktywacji i technik stabilizacji na proces treningu.

### 3. Zbiór Danych: Predykcja Churnu Klientów Banku

Do projektu wykorzystamy powszechnie dostępny i dobrze zdefiniowany zbiór danych z platformy Kaggle:

* **Nazwa:** **Bank Customer Churn Prediction**
* **Link:** [https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
* **Opis:** Zbiór zawiera dane demograficzne i transakcyjne klientów banku.
* **Zadanie:** Przewidzieć, czy klient zrezygnuje z usług banku (kolumna `Exited` = 1) czy nie (`Exited` = 0). 

### 4. Wymagania Projektowe (Kroki)

Aby projekt został zaliczony, musi zawierać implementację **wszystkich** poniższych punktów, odzwierciedlających materiał z prezentacji:

#### Etap I: Struktura OOP i Przetwarzanie Danych

1.  **Własna Klasa `Dataset`:**
    * Należy stworzyć własną klasę (np. `ChurnDataset`) dziedziczącą po `torch.utils.data.Dataset`.
    * W metodzie `__init__` należy wczytać dane (np. z pliku CSV), dokonać **niezbędnego preprocessingu** (patrz: Wskazówki Wykładowcy) i przechować je (np. jako tensory lub tablice NumPy).
    * Metody `__len__` i `__getitem__` muszą być poprawnie zaimplementowane.
2.  **`DataLoader`:**
    * Należy utworzyć instancje `DataLoader` dla zbiorów treningowego i testowego.
    * Dla danych treningowych należy obowiązkowo ustawić `shuffle=True` oraz wybrać odpowiedni `batch_size`.
3.  **Własna Klasa Modelu `nn.Module`:**
    * Należy zdefiniować architekturę sieci (MLP) jako klasę dziedziczącą po `nn.Module`.
    * W metodzie `__init__` należy zdefiniować wszystkie warstwy (np. `nn.Linear`, `nn.BatchNorm1d`).
    * W metodzie `forward` należy zdefiniować przepływ danych (forward pass).
    * Model powinien mieć co najmniej 3 warstwy ukryte.

#### Etap II: Pętla Treningowa i Ewaluacja

1.  **Pętla Treningowa:**
    * Należy zaimplementować pełną pętlę treningową, iterującą po epokach i batchach z `DataLoader`'a.
    * Pętla musi zawierać wszystkie kluczowe kroki: zerowanie gradientów (`optimizer.zero_grad()`), forward pass, obliczenie straty, backward pass (`loss.backward()`) oraz aktualizację wag (`optimizer.step()`).
2.  **Funkcja Straty i Metryka:**
    * Jako funkcję straty należy użyć `nn.BCELoss` (Binary Cross-Entropy), adekwatnie do zadania i przykładu z prezentacji.
    * Do ewaluacji należy wykorzystać `torchmetrics.Accuracy` z `task="binary"`.
3.  **Tryb Ewaluacji:**
    * Należy zaimplementować osobną funkcję lub pętlę do ewaluacji modelu na zbiorze testowym.
    * Musi ona zawierać przełączenie modelu w tryb ewaluacji (`net.eval()`) oraz blok `with torch.no_grad()`.

#### Etap III: Eksperymenty ze Stabilizacją i Optymalizacją

To jest kluczowa, badawcza część projektu. Należy porównać kilka wariantów modelu:

1.  **Model Bazowy (Niestabilny):**
    * Stwórz model używający domyślnej inicjalizacji wag oraz funkcji aktywacji `nn.ReLU`.
    * Nie używaj normalizacji batchowej.
2.  **Porównanie Technik Stabilizacji (jako osobne eksperymenty):**
    * **Eksperyment A (Inicjalizacja):** Zastosuj inicjalizację He/Kaiming (`init.kaiming_uniform_`) do warstw `nn.Linear`. Porównaj wyniki z Modelem Bazowym.
    * **Eksperyment B (Funkcje Aktywacji):** Zastąp `nn.ReLU` funkcją `nn.ELU`. Porównaj wyniki z Modelem Bazowym (i opcjonalnie z A).
    * **Eksperyment C (Batch Normalization):** Dodaj warstwy `nn.BatchNorm1d` po każdej warstwie liniowej, a przed funkcją aktywacji. Porównaj wyniki z pozostałymi.
3.  **Porównanie Optymalizatorów:**
    * Wybierz **najlepszą architekturę** z poprzedniego kroku (prawdopodobnie łączącą wszystkie techniki stabilizacji).
    * Wytrenuj ten sam model, używając co najmniej trzech różnych optymalizatorów: **SGD**, **RMSprop** i **Adam**.
    * Porównaj ich szybkość zbieżności (krzywe straty) i finalną dokładność.

---

### 5. Sposób Ewaluacji (Oceny)

Ocenie podlegać będzie **raport** (w formie notatnika Jupyter Notebook lub PDF) oraz kod źródłowy. Raport musi zawierać:

1.  Opis przeprowadzonego preprocessingu danych.
2.  Jasną implementację wszystkich klas (`Dataset`, `Model`).
3.  **Wizualizacje (wykresy):**
    * Krzywe straty (training loss) i dokładności (validation accuracy) dla każdego z eksperymentów ze stabilizacji (Etap III, pkt 2).
    * Krzywe straty (training loss) dla porównywanych optymalizatorów (Etap III, pkt 3).
4.  Tabelę zbiorczą prezentującą finalną dokładność na zbiorze testowym dla wszystkich testowanych wariantów.
5.  **Wnioski:** Krótka analiza (kilka zdań) odpowiadająca na pytania:
    * Która technika stabilizacji miała największy wpływ na wyniki?
    * Który optymalizator okazał się najlepszy dla tego problemu i dlaczego?
    * Jaki był najlepszy osiągnięty wynik?

---

### 6. Oczekiwane Rezultaty

* **Wymagania Minimalne :**
    * Kod jest działający i kompletny (implementuje Etapy I i II).
    * Przeprowadzono przynajmniej jeden eksperyment z Etapu III (np. porównanie optymalizatorów LUB porównanie stabilizacji).
    * Raport zawiera podstawowe wyniki, ale może brakować głębszej analizy lub wizualizacji.
    * Osiągnięta dokładność (Accuracy) na zbiorze testowym > **82%**.

* **Wymagania Zadowalająca :**
    * Spełnione wszystkie wymagania projektowe (Etapy I, II, III).
    * Kod jest czysty i dobrze zorganizowany.
    * Raport zawiera wszystkie wymagane wizualizacje, tabele i klarowne wnioski.
    * Osiągnięta dokładność (Accuracy) > **86%** ORAZ raport zawiera wnikliwą analizę wyników i błędów modelu (np. analiza macierzy pomyłek).

**@ Plan projektu został wygenerowany przez sztuczną inteligencje**