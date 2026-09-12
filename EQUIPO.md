# Cómo trabajamos los cuatro

## Lo primero que tienen que saber

Ya tenemos un programa que funciona y que acierta casi siempre. Eso ya está ganado.
Todo lo que sigue es para mejorarlo, **no para rehacerlo**.

La regla más importante del equipo:

> **Nunca dejes el proyecto en un estado donde no prende.**

Si a las 3 de la mañana alguien rompe algo y nadie lo arregla, a la hora de presentar
no tenemos nada que enseñar. Ese es el único error que no se puede deshacer.

---

## Antes de empezar (todos, una vez)

```bash
git clone https://github.com/0xkdani/RetoAltur-HackMTY26.git
cd RetoAltur-HackMTY26

python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tests\smoke_test.py
```

Si la última línea dice **11/11 verificaciones pasaron**, ya estás listo.
Funciona igual con Python 3.12, 3.13 o 3.14.

Para levantar el programa y verlo funcionando:

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Y en otra ventana, para comprobar que responde:

```bash
curl http://127.0.0.1:8000/health
```

### Si vas a entrenar o medir, necesitas los audios

Pídeselos a Daniel. Son 671 MB y **no se suben al repositorio** porque Altur no
permite redistribuirlos. Las instrucciones para conectarlos están en el `README.md`,
en la sección "Puesta a punto del dataset".

---

## Las dos reglas para no estorbarnos

**1. Cada quien trabaja en su propia rama.**

```bash
git checkout -b mi-encargo      # el nombre de tu encargo: palabras, oido, romper, demo
```

**2. Antes de juntar tu trabajo, corre la prueba.**

```bash
.venv\Scripts\python.exe tests\smoke_test.py
```

Si sale en rojo, **no lo juntes**. Avisa y lo vemos.

Daniel es quien junta todo. Así nadie mete algo roto sin querer.

**Nadie toca `app/fusion.py` ni `app/detector.py` sin avisar al grupo.** Esos dos
archivos son los que amarran todo; si dos personas los cambian a la vez, se pelean.

Nos juntamos 10 minutos cada 3 o 4 horas: qué llevas, en qué estás atorado,
seguimos o lo cortamos.

---

# Los cuatro encargos

Nadie necesita ser experto. Tu papel es **ser el dueño de un área**: decidir qué se
intenta, comprobar si sirvió, y avisar si no. Si te atoras, puedes abrir Claude Code
en la carpeta del proyecto y pedirle ayuda — abajo te dejo qué pedirle para arrancar.

---

## Encargo 1 — Las palabras

**Qué logras:** hoy el programa no entiende lo que se dice, solo los ritmos y los
sonidos. Tú haces que además lea las palabras.

**Por qué importa:** el agente del banco a veces pregunta por cosas que no existen.
Una persona real contesta "no tengo eso". Un robot tiende a inventarse una respuesta
con tal de contestar algo. Esa diferencia se puede detectar y sería nuestra señal
más original.

**Tu archivo:** `app/features/semantic.py` (lo creas tú, nadie más lo toca).

**Cómo sabes que funcionó:** corres `python scripts/train_fusion.py` y miras el
renglón que dice `fusión` en la sección de validación cruzada. Ahorita el número de
EER es **0.028**. Si con lo tuyo baja, sirve. Si no baja, no entra.

**Cuidado:** leer las palabras es lento. El programa hoy tarda 200 milisegundos y la
rapidez es uno de los criterios del jurado. Tu parte debe correr solo cuando el
programa no está seguro, no siempre.

**Para arrancar, pídele a Claude Code:**
> "Lee el README y créame el archivo app/features/semantic.py usando faster-whisper.
> Debe transcribir los dos canales, alinear los turnos con los que da el VAD, y sacar
> features de: respuestas a preguntas del agente sobre cosas que no existen,
> muletillas reales del español ('este', 'eh', 'o sea'), y auto-correcciones. Sigue
> el mismo contrato que app/features/timing.py."

---

## Encargo 2 — El oído

**Qué logras:** mejorar la parte que escucha *cómo suena* la voz.

**Por qué importa:** hoy esa parte es una versión sencilla que hicimos rápido. Ya
funciona bien, pero seguro se puede mejorar, y es la que más se apoya en lo que se
usa en la industria.

**Tu archivo:** `app/features/acoustic.py`.

**Cómo sabes que funcionó:** corres `python scripts/train_fusion.py` y miras el
renglón `solo acústica`. Ahorita está en **0.064**. Tu objetivo es bajarlo.

**Cuidado:** las llamadas son de teléfono, de baja calidad. Casi todo lo que
encuentres en internet está hecho para audio de buena calidad y aquí se comporta
peor de lo que promete. No te sorprendas si un modelo famoso rinde mal.

**Cuidado 2:** la tarjeta de video de Daniel tiene 4 GB. Los modelos grandes no caben.

**Para arrancar, pídele a Claude Code:**
> "Lee el README y ayúdame a mejorar app/features/acoustic.py. Quiero comparar el
> baseline actual contra un modelo más fuerte, midiendo siempre con
> scripts/train_fusion.py y respetando el split del manifiesto. Ten en cuenta que el
> audio es telefónico de 8 kHz y que solo tengo 4 GB de VRAM."

---

## Encargo 3 — El que rompe

**Qué logras:** encontrar en qué casos el programa se equivoca, antes de que lo
encuentre el jurado.

**Por qué importa:** este encargo es el que más puntos da. El jurado califica
"robustez", y la única forma de demostrarla es enseñando que ya buscaste dónde falla.

**Tu archivo:** `scripts/diagnose.py`.

**Por dónde empezar, en este orden:**

1. **Escucha las 3 llamadas que falla.** Son sintéticas que el programa cree humanas.
   Una de ellas la falla con mucha seguridad, que es lo preocupante. Escúchalas
   completas y anota qué tienen de raro.
2. **Ensúciale el audio.** Métele ruido, bájale el volumen, córtale pedazos.
   ¿A partir de cuánto se rompe?
3. **Simula un atacante listo.** Nuestra mejor señal es que los robots contestan
   con tiempos muy parejos. ¿Qué pasa si un atacante le mete variación al azar a
   sus tiempos de respuesta? Eso es lo primero que preguntaría un juez.

**Cómo sabes que funcionó:** cada cosa que encuentres se convierte en un renglón de
la tabla del `README.md`. Esa tabla es nuestro mejor argumento el día de presentar.

**Para arrancar, pídele a Claude Code:**
> "Lee el README y scripts/diagnose.py. Quiero (1) sacar a una carpeta las llamadas
> que el modelo falla para escucharlas, (2) una prueba que degrade el audio a
> propósito y mida cuánto aguanta, y (3) una prueba que simule un atacante que
> aleatoriza sus tiempos de respuesta para evadir la señal de timing."

---

## Encargo 4 — El que prende y lo cuenta

**Qué logras:** que el día del jurado todo funcione, y que la presentación esté lista.

**Por qué importa:** el jurado nos visita **15 minutos** y prueba nuestro programa en
vivo. Si la conexión se cae o la computadora se durmió, no importa nada de lo demás.
Este encargo parece el aburrido y es el que puede costarnos todo.

**Tus archivos:** `DEPLOY.md` y la presentación.

**Tus tareas:**

1. **Haz que el programa se pueda usar desde fuera.** Las instrucciones están en
   `DEPLOY.md`. Pruébalo desde el celular con datos, sin wifi, para confirmar que
   de verdad funciona desde afuera.
2. **Apaga la suspensión de la computadora.** En serio. Pasa siempre.
3. **Ten un plan B.** Si la conexión del evento falla, qué hacemos.
4. **Arma la presentación.** Reparto sugerido de los 15 minutos: 2 minutos el
   problema, 3 minutos enseñándolo funcionando, 5 minutos de resultados, 5 de
   preguntas.
5. **Ensáyenlo completo con cronómetro**, al menos una vez, todos juntos.
   Siempre nos vamos a pasar de tiempo.

**Prepara la respuesta a esta pregunta**, porque nos la van a hacer:
*"¿cómo saben que su programa no está haciendo trampa?"*

Ya está contestada con números en el `README.md`, en la tabla de "¿Aprende la voz o
aprende el canal?". Apréndetela. Muy pocos equipos van a poder responder eso.

Y la otra que nos van a hacer: *"les salió 100% de acierto, ¿no están exagerando?"*.
La respuesta honesta es que sí acertamos 71 de 71, pero son pocas llamadas para
presumir un 100%; el número que reportamos es el de la prueba más estricta, **0.028**.
Decirlo así nos da credibilidad en vez de quitárnosla.

---

## Cuándo cortar algo

Esta es la parte que casi nadie hace y la que más sirve:

> **Si faltando 6 horas tu encargo no está funcionando mejor que lo que ya teníamos,
> se abandona** y te pasas a ayudar con la presentación.

El reto dice, con todas sus letras, que prefieren **una cosa bien hecha que tres a
medias**. Ya tenemos una cosa bien hecha. Lo demás es extra, y un encargo a medias no
suma nada y sí puede romper lo que ya funciona.
