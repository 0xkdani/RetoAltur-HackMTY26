# Exponer el endpoint durante el judging

El brief solo pide que `POST /detect` sea alcanzable durante la ventana de
evaluación. La opción más rápida y sin Docker: correrlo local y sacarlo por un
túnel.

## Opción A — túnel de Cloudflare (recomendada, sin cuenta)

```bash
# 1. Levanta el servicio
bash run.sh

# 2. En otra terminal, saca el túnel
winget install --id Cloudflare.cloudflared     # una sola vez
cloudflared tunnel --url http://localhost:8000
```

Imprime una URL pública `https://algo-random.trycloudflare.com`.
El endpoint del scorer sería `https://algo-random.trycloudflare.com/detect`.

Verifica antes de entregar la URL:

```bash
curl https://TU-URL.trycloudflare.com/health
```

## Opción B — ngrok

```bash
ngrok http 8000
```

Requiere cuenta gratuita y token. La URL cambia en cada reinicio igual que
Cloudflare, salvo que pagues dominio fijo.

## Checklist antes de que lleguen los jueces

- [ ] `python tests/smoke_test.py --http` pasa contra la URL pública
- [ ] `/health` responde `model_trained: true` (si ya entrenaron)
- [ ] La laptop no se suspende — desactiva el suspensión por inactividad
- [ ] El túnel lleva rato levantado sin caerse
- [ ] Ten una segunda terminal lista para relevantar si se cae

## Riesgo conocido

El túnel depende de la red del evento y de que la laptop siga despierta. Si la
conexión del venue es mala, el plan B es desplegar en un servicio gestionado
(Render, Railway o Modal aceptan el repo tal cual: `pip install -r
requirements.txt` y `uvicorn app.main:app`). El modelo entrenado pesa poco,
así que `models/fusion.joblib` puede ir en el repo para ese caso — quítenlo del
`.gitignore` si toman esa ruta.
