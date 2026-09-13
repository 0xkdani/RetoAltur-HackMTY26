# Arquitectura

Dos mundos separados, y conviene no confundirlos:

- **En vivo** — lo que pasa cada vez que llega una llamada. Corre en 200 ms.
- **De antemano** — el entrenamiento. Se hace una vez y deja un archivo de 4 KB.

---

## 1. El recorrido de una llamada (en vivo)

```mermaid
flowchart TD
    J([Jurado]) -->|"POST /detect<br/>WAV en base64"| M

    subgraph API["app/main.py"]
        M["Recibe y busca el audio<br/>en el payload"]
    end

    M --> D["app/detector.py<br/><i>coordina todo</i>"]

    D --> A["app/audio.py<br/>separa las 2 pistas"]
    A -->|"pista 0: caller<br/>pista 1: agente"| V

    V["app/vad.py<br/>¿cuándo habla cada quien?"]
    V -->|"turnos de los dos"| T
    V -->|"turnos del caller"| AC

    subgraph F["app/features/"]
        T["timing.py<br/><b>23 medidas</b><br/>tiempos de respuesta,<br/>interrupciones, pausas"]
        AC["acoustic.py<br/><b>19 medidas</b><br/>timbre, ritmo, tono"]
    end

    T --> FU
    AC --> FU

    FU["app/fusion.py<br/><b>42 medidas → una probabilidad</b>"]
    MOD[("models/fusion.joblib<br/>4 KB")] -.->|pesos aprendidos| FU

    FU --> R["is_synthetic + confidence"]
    R --> LOG[("logs/consultas.jsonl")]
    R -->|respuesta JSON| J

    style J fill:#e8e8e8,stroke:#666
    style FU fill:#d4e6f1,stroke:#2874a6
    style T fill:#d5f5e3,stroke:#1e8449
    style MOD fill:#fdebd0,stroke:#b9770e
```

**Lo importante del diagrama:** `vad.py` alimenta a los dos bloques de medidas,
pero `timing.py` necesita los turnos de **ambas** pistas y `acoustic.py` solo los
del caller. Esa diferencia es la idea central del proyecto: sin el canal del
agente, "respondió en 900 ms" no significa nada.

---

## 2. El entrenamiento (de antemano, una sola vez)

```mermaid
flowchart LR
    subgraph DATOS["data/ (no se versiona)"]
        MAN[["manifest.csv<br/>353 llamadas etiquetadas"]]
        AUD[["audio/*.wav"]]
        TUR[["turns/*.json<br/>de referencia"]]
    end

    MAN --> DS["scripts/dataset.py<br/>junta audio + etiqueta + grupo"]
    AUD --> DS
    TUR -.->|"solo para comparar"| DS

    DS --> EX["scripts/extract_features.py<br/>42 medidas × 353 llamadas<br/><i>60 ms cada una</i>"]
    EX --> CSV[["data/features_vad.csv"]]

    CSV --> TR["scripts/train_fusion.py<br/>entrena con 'train'<br/>mide con 'val'"]
    TR --> MOD[("models/fusion.joblib")]

    CSV --> DG["scripts/diagnose.py<br/>¿hay trampa?"]
    MOD --> EV["scripts/eval_endpoint.py<br/>prueba por HTTP"]

    style MOD fill:#fdebd0,stroke:#b9770e
    style DG fill:#fadbd8,stroke:#a93226
```

`diagnose.py` no produce nada que el programa use: produce **evidencia**. Es el
que contesta "¿aprende la voz o aprende el canal?" cuando el jurado pregunte.

---

## 3. Quién depende de quién

```mermaid
flowchart TD
    main["main.py<br/><i>la puerta</i>"] --> detector["detector.py"]
    main --> registro["registro.py"]
    main --> audio1["audio.py"]
    main --> fusion1["fusion.py"]

    detector --> audio["audio.py"]
    detector --> vad["vad.py"]
    detector --> feat["features/"]
    detector --> fusion["fusion.py"]

    feat --> timing["timing.py"]
    feat --> acoustic["acoustic.py"]
    timing --> vad
    acoustic --> vad

    style main fill:#d4e6f1,stroke:#2874a6
    style vad fill:#d5f5e3,stroke:#1e8449
```

`vad.py` y `audio.py` **no dependen de nada interno**. Eso es a propósito: son
las piezas de abajo y se pueden probar solas. Si mañana cambian el detector de
voz por silero, nada más se toca ese archivo.

---

## 4. La cascada (pensada para la latencia)

La latencia es criterio de calificación, así que el sistema está diseñado para
no gastar tiempo cuando no hace falta.

```
  Llamada
     │
     ▼
  ┌─────────────────────────────────────┐
  │ NIVEL 1  ·  ~200 ms  ·  siempre     │   ← implementado
  │ turnos + timing + acústica barata   │
  └─────────────────────────────────────┘
     │
     ├── ¿seguro?  ──── sí ──►  responder
     │
     ▼ no (zona gris)
  ┌─────────────────────────────────────┐
  │ NIVEL 2  ·  ~2 s   ·  solo si dudó  │   ← pendiente (encargo 2)
  │ modelo acústico pesado              │
  └─────────────────────────────────────┘
     │
     ▼ sigue dudando
  ┌─────────────────────────────────────┐
  │ NIVEL 3  ·  ~10 s  ·  casi nunca    │   ← pendiente (encargo 1)
  │ transcribir y buscar respuestas     │
  │ inventadas                          │
  └─────────────────────────────────────┘
```

Hoy **casi todas las llamadas se resuelven en el nivel 1**. El andamiaje de los
otros dos ya está puesto en `detector.py`.

---

## 5. Las puertas del servicio

| Puerta | Quién la usa | Para qué |
|---|---|---|
| `POST /detect` | El jurado, automático | **La que califica.** Recibe el WAV, devuelve el veredicto |
| `GET /stats` | Ustedes, durante el judging | ¿Ya empezaron? ¿cuántas llevan? ¿truena algo? |
| `GET /health` | Ustedes y el túnel | ¿Está vivo? ¿tiene modelo cargado? |
| `POST /detect/upload` | Ustedes, a mano | Subir un archivo para probar |

---

## Tres decisiones que se ven en los diagramas

**El modelo es un archivo, no código.** `fusion.joblib` pesa 4 KB y se versiona.
Por eso el servicio arranca recién clonado sin necesitar los 671 MB de audio.

**El audio nunca se guarda.** `registro.py` anota cuándo llegó, cuánto pesaba y
qué respondimos — nunca el contenido. Son voces de personas reales.

**Entrenamos con lo mismo que servimos.** El dataset trae turnos de referencia
mejores que los nuestros, pero el jurado solo manda el WAV. Si entrenáramos con
la referencia, habría un desajuste invisible entre la prueba y la realidad. Por
eso `turns/` aparece punteado en el diagrama: se usa para comparar, no para
entrenar el modelo que se sirve.
