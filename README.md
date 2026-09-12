# Altur VoiceGuard

Detección de voz sintética en llamadas telefónicas.
Reto **Defend the Bank Against Voice Deepfakes** — HackMTY 2026.

`POST /detect` recibe un WAV estéreo de 8 kHz (canal 0 = caller, canal 1 = agente)
y responde si quien llama es una voz sintética.

```json
{ "is_synthetic": true, "confidence": 0.87 }
```

---

## El enfoque

La apuesta central no es el audio, es **cómo reacciona el caller a la conversación**.

Un humano y un pipeline de voz sintética no responden igual a las interrupciones.
El humano responde con latencias caóticas —a veces se encima a los 150 ms, a veces
se queda pensando dos segundos—, interrumpe al agente, se apaga a medias cuando lo
atropellan y suelta un "¿bueno?" cuando el agente se calla. Un bot paga siempre el
mismo costo de ASR + LLM + TTS, así que responde más lento y, sobre todo, **más
consistente**. Esa consistencia es la delación.

Los datos le dan la razón a la hipótesis: de las 42 features del sistema, las dos
que más pesan en el modelo son la **mediana de la latencia de respuesta** (+1.35) y
el **coeficiente de variación de esa latencia** (−0.89). La segunda es la más
interesante porque mide dispersión, no magnitud: así no confunde "bot lento" con
"bot consistente". Un caller con latencias parejas es sospechoso aunque sean rápidas.

Esto solo es posible usando **los dos canales**. El canal del agente nos dice a qué
estaba reaccionando el caller: sin él, "respondió en 900 ms" no significa nada.

### Las tres señales

| Señal | Qué mide | Estado |
|---|---|---|
| **B. Conversacional** | Latencias de respuesta, barge-in, recuperación tras interrupción, sondeos en silencio | 23 features |
| **A. Acústica** | Planitud espectral, ritmo silábico, variabilidad de F0, flujo espectral | 19 features |
| **C. Semántica** | Preguntas trampa del agente: el humano dice "no tengo eso", el LLM inventa | Pendiente |

Se combinan con **fusión tardía**: cada bloque produce su vector y una regresión
logística encima los junta. Es a propósito un modelo simple — se entrena con 282
llamadas sin sobreajustar, y sus coeficientes son explicables frente a un banco.
La `confidence` sale de la distancia al umbral, no de un número inventado.

---

## Resultados

Dataset del reto: 353 llamadas (203 sintéticas, 150 humanas), split
speaker-disjoint del manifiesto de Altur.

**Validación cruzada dentro de train** (5 folds, out-of-fold, 282 llamadas):

| Modelo | AUC | EER | Brier |
|---|---|---|---|
| Solo timing | 0.970 | 0.092 | 0.070 |
| Solo acústica | 0.981 | 0.064 | 0.047 |
| **Fusión** | **0.996** | **0.028** | **0.022** |

**Set de validación**, hablantes que no aparecen en train (71 llamadas):

| Modelo | AUC | EER |
|---|---|---|
| Solo timing | 0.998 | 0.014 |
| Solo acústica | 0.999 | 0.028 |
| **Fusión** | **1.000** | **0.000** |

**Endpoint HTTP contra val**, con un modelo entrenado *solo* con train:
**71/71 correctas**, latencia media **203 ms**, p95 **277 ms** por llamada de ~2.5
minutos, en CPU.

> **Honestidad sobre el 0.000.** Val son 71 llamadas. Cero errores en 71 intentos no
> significa que el EER verdadero sea cero: el intervalo de confianza al 95% llega
> hasta ~4-5% de error. El número que reportamos como estimación es el de la
> validación cruzada (**EER 0.028**), no el de val. El set oculto del judging usa
> voces que no aparecen en ninguno de los dos splits, así que esperamos degradación.

### ¿Aprende la voz o aprende el canal?

Un EER de 0.00 es sospechoso hasta que se demuestre lo contrario.
`scripts/diagnose.py` corre las pruebas de control:

| Prueba | Resultado | Lectura |
|---|---|---|
| `duration_s` sola | AUC 0.531 | No hay sesgo de duración entre clases |
| `noise_floor_db` sola | AUC 0.568 | El ruido de fondo **no** separa las clases |
| Sin ninguna feature de canal (35 de 42) | AUC 1.000 | No dependemos del setup de grabación |
| Sin el top-5 de features individuales | AUC 1.000 | La señal está distribuida, no hay atajo |
| Solo features conductuales (23) | AUC 0.998 | El timing solo ya resuelve casi todo |

La última fila es la que más importa: las features conductuales no tocan la forma de
onda de la voz, solo **cuándo** se habla. Son las más difíciles de falsificar para un
atacante, porque exigirían rehacer la arquitectura del pipeline de voz, no cambiar de
modelo TTS.

---

## Correr el proyecto

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/Mac

./.venv/Scripts/python.exe tests/smoke_test.py    # 11 verificaciones end-to-end
bash run.sh                                       # servicio en :8000
```

El modelo entrenado (`models/fusion.joblib`, 4 KB) está versionado, así que el
servicio funciona recién clonado sin necesidad del dataset.

### Puesta a punto del dataset (solo si vas a reentrenar)

El dataset **no se versiona**: los términos de Altur dicen que no se redistribuye.
Hay que enlazarlo localmente.

```powershell
# 1. manifest.csv y turns/ viven en el repo del reto, NO en el ZIP del release
git clone https://github.com/alturio/hackmty26.git repo
Copy-Item repo/manifest.csv data/manifest.csv

# 2. Bajar altur-challenge-audio.zip del release v1.0 y descomprimirlo donde sea

# 3. Enlazar con junctions (no requieren permisos de administrador)
New-Item -ItemType Junction -Path data/audio -Target "C:/ruta/a/altur-challenge-audio/audio"
New-Item -ItemType Junction -Path data/turns -Target "$PWD/repo/turns"
```

Debe quedar así:

```
data/manifest.csv      anon_id,label,split,duration_s
data/audio/<id>.wav    353 llamadas, estéreo 8 kHz
data/turns/<id>.json   segmentos de voz por canal, de referencia
```

### Reproducir los resultados

```bash
python scripts/ingest.py                        # formatos, balance, duraciones
python scripts/extract_features.py              # features con nuestro VAD
python scripts/extract_features.py --turns      # con los turnos de referencia
python scripts/train_fusion.py                  # entrena, calibra, reporta CV y val
python scripts/diagnose.py                      # robustez y ablación
python scripts/eval_endpoint.py --split val     # benchmark contra el endpoint vivo
```

`train_fusion.py --train-only` entrena sin el split de validación, que es lo que hay
que usar para medir el endpoint contra val sin contaminar el resultado.

---

## Decisiones que importan

**El split del manifiesto, no uno propio.** Altur lo construyó speaker-disjoint. Un
split aleatorio dejaría al mismo hablante en train y test e inflaría la métrica. El
script también corre `GroupKFold` dentro de train, porque 71 llamadas de val son
pocas para un número estable.

**Entrenamos con la segmentación que usamos en producción.** El dataset trae turnos
precalculados en `turns/`, y son mejores que los nuestros: con ellos el timing solo
pasa de EER 0.092 a 0.049 en CV. Pero el scorer solo manda el WAV, así que el modelo
que se sirve está entrenado con los segmentos de **nuestro** VAD. Entrenar con la
referencia habría creado un mismatch invisible entre entrenamiento y producción.

**El 8 kHz no es un detalle, es el problema.** Casi toda la literatura de
anti-spoofing viene de audio 16 kHz limpio, donde los artefactos de vocoder viven
arriba de 4 kHz. La telefonía los borra: recorta a 300–3400 Hz y mete G.711. Por eso
`scripts/telephony_augment.sh` pasa cualquier dataset externo por ese canal antes de
usarlo para aumentar el set.

**Warm-up al arrancar.** La primera inferencia de un proceso paga la carga perezosa
de joblib/sklearn: ~1.2 s en frío contra ~25 ms en caliente. El servicio corre una
inferencia dummy al arrancar para que el primer request real —que puede ser el que
cronometren— no la pague.

**El endpoint no se cae y no rechaza payloads.** El brief no fija el nombre del campo
del base64, así que aceptamos diez variantes, JSON sin header, WAV crudo en el body y
multipart. Si algo revienta internamente responde 200 con el veredicto neutro y el
error en `meta`. Un 500 durante el scoring es cero puntos.

---

## Estructura

```
app/
  main.py              API FastAPI, POST /detect, warm-up de arranque
  detector.py          orquestación en cascada
  audio.py             decodificación base64/WAV, separación de canales
  vad.py               VAD de energía + envoltorio opcional de silero
  fusion.py            regresión logística + calibración de confidence
  features/
    timing.py          señal B: 23 features conversacionales
    acoustic.py        señal A: 19 features espectrales
scripts/
  ingest.py            exploración del dataset
  extract_features.py  dataset a CSV (multiproceso, 60 ms por llamada)
  train_fusion.py      entrenamiento, split oficial, EER y Brier
  diagnose.py          ablación y pruebas de artefacto de canal
  eval_endpoint.py     benchmark HTTP como lo hará el scorer
  telephony_augment.sh simulación del canal telefónico con ffmpeg
tests/
  make_fixture.py      llamadas sintéticas con turnos controlados
  smoke_test.py        11 verificaciones end-to-end
models/
  fusion.joblib        modelo entrenado (versionado, 4 KB)
```

## Estado

- [x] Endpoint con el contrato del reto, 11/11 verificaciones
- [x] Señal conversacional y baseline acústico, EER 0.028 en CV
- [x] Pipeline reproducible: ingesta, features, entrenamiento, diagnóstico
- [x] Robustez verificada: sin atajos de canal ni de duración
- [x] 203 ms por llamada de 2.5 min, en CPU, sin GPU
- [ ] Señal semántica con Whisper para el paso 3 de la cascada
- [ ] Modelo acústico fuerte (AASIST / wav2vec2) para el paso 2
