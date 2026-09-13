# Desplegar el servicio en Render

El reto solo pide que `POST /detect` sea alcanzable durante la ventana de
evaluación. Lo desplegamos en Render para que sea una URL real, que no dependa
de que una laptop siga prendida ni de la wifi del evento.

---

## Por qué Render y no Vercel

Vercel no sirve para este proyecto, y no es cuestión de gusto:

| | Lo que necesitamos | Lo que Vercel permite |
|---|---|---|
| Tamaño de cada petición | **6.3 MB** de media, hasta 11.7 MB | **4.5 MB** |
| Peso de las librerías | **349 MB** | **250 MB** |

Medido sobre las 353 llamadas del dataset: **332 de 353 (el 94%) exceden el
límite de Vercel**. Las rechazaría con error 413 antes de llegar a nuestro
código. Ese tope es de la plataforma, no del plan: pagar no lo sube.

Vercel está hecho para webs y APIs ligeras. Nosotros recibimos audio de varios
megas y cargamos librerías científicas pesadas.

---

## Pasos para desplegar

### 1. Crear la cuenta

Entra a [render.com](https://render.com) y regístrate **con tu cuenta de
GitHub**. Así Render ve tus repositorios directamente.

### 2. Crear el servicio

1. Botón **New +** → **Web Service**
2. Elige el repositorio `RetoAltur-HackMTY26`
3. Render detecta el archivo `render.yaml` y llena todo solo

Si no lo detecta, ponlo a mano:

| Campo | Valor |
|---|---|
| Language | `Python 3` |
| Build Command | `pip install -r requirements-prod.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | `Free` |
| Health Check Path | `/health` |

4. **Create Web Service**

El primer despliegue tarda entre 5 y 10 minutos, casi todo instalando
scikit-learn y numpy.

### 3. Comprobar que quedó

Render te da una URL tipo `https://altur-voiceguard.onrender.com`.

```bash
curl https://TU-URL.onrender.com/health
```

Debe responder `{"status":"ok","model_trained":true}`.

Si `model_trained` sale en `false`, el modelo no se subió al repo. Revisa que
`models/fusion.joblib` esté versionado.

### 4. Probarlo con llamadas de verdad

Desde tu computadora, apuntando a la URL pública:

```bash
.venv\Scripts\python.exe scripts\eval_endpoint.py --split val --url https://TU-URL.onrender.com/detect
```

**Esta es la única prueba que cuenta.** Debe dar 71/71.

### 5. Activar el ping automático

El plan gratuito duerme el servicio tras 15 minutos sin tráfico, y la siguiente
petición tarda **cerca de un minuto** en responder mientras despierta. Si esa
petición es la primera del jurado, quedamos mal sin merecerlo.

Edita `.github/workflows/keep-alive.yml` y pon tu URL donde dice
`PON_AQUI_TU_URL`. Súbelo y GitHub hará ping cada 10 minutos, gratis.

---

## Qué cambia al estar en Render

**El registro se borra al reiniciar.** Render no guarda archivos entre
reinicios, así que `logs/consultas.jsonl` y el contador de `/stats` se vacían
cada vez que el servicio se reinicia o despierta. Durante el judging funciona
perfecto, que es cuando importa.

**No hace falta el dataset.** El modelo entrenado (4 KB) está versionado en el
repo, así que el servicio arranca sin los 671 MB de audio.

**Sin pandas.** `requirements-prod.txt` no lo incluye: el servicio no lo usa,
solo los scripts de entrenamiento. Ahorra memoria en un plan de 512 MB.
Verificado: el servicio arranca y responde sin pandas instalado.

---

## Checklist del día del judging

- [ ] `curl https://TU-URL.onrender.com/health` responde `200`
- [ ] `eval_endpoint.py` contra la URL pública da 71/71
- [ ] El ping automático está corriendo (pestaña **Actions** del repo, en verde)
- [ ] La pestaña `/stats` abierta para ver entrar las consultas del jurado
- [ ] La URL escrita en un papel, por si hay que dictarla
- [ ] Alguien sabe **dónde** hay que entregar la URL y **a qué hora** los visitan

---

## Si algo falla el día del evento

**El servicio no responde** → entra al panel de Render, pestaña *Logs*. Si dice
"out of memory", el plan gratuito se quedó corto: cambia a instancia de pago por
unas horas, son unos pocos dólares.

**Responde lento la primera vez** → estaba dormido. Haz un `curl` al `/health`
diez minutos antes de que lleguen los jueces.

**Render se cae del todo** → plan B: levanta el servicio en tu laptop
(`bash run.sh`) y expónlo con un túnel:

```bash
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

Ten `cloudflared` **ya instalado desde antes**. Instalarlo con los jueces
enfrente es la peor forma de gastar los 15 minutos.
