# Lo que les puede preguntar el jurado

Regla de oro: **si no lo midieron, la respuesta es "no lo medimos"**, y si acaso,
cómo lo medirían. Un número inventado que se caiga en la siguiente pregunta
destruye la credibilidad de todos los números buenos.

---

## Las cinco que casi seguro les hacen

### 1. "¿Cómo saben que no están sobreajustando?"

> "Intentamos romperlo nosotros. Le quitamos las 7 medidas de canal y rinde
> igual. Le quitamos sus 5 medidas más fuertes y rinde igual. Comprobamos si
> la duración de la llamada sola separaba las clases: AUC 0.53, o sea nada. Y
> si había aprendido el ruido del micrófono: 0.57. La señal está repartida
> entre las 42, no hay un atajo del que dependamos."

### 2. "¿Y si el atacante aleatoriza sus tiempos de respuesta?"

**La mejor pregunta que les pueden hacer. No la esquiven.**

> "Es el ataque obvio y nos rompería la señal principal. Tres cosas: primero,
> no es gratis — esperar de más cuesta tiempo y llamadas en una campaña a
> escala. Segundo, no basta con ruido: tendría que imitar la forma de la
> distribución humana, que tiene cola larga; el azar uniforme también es un
> patrón detectable. Tercero, no dependemos de una sola señal: la parte
> acústica sola da 6.4% de error y esa no se evade cambiando tiempos.
>
> Con más tiempo, ese es exactamente el primer experimento que haríamos."

### 3. "¿Por qué no usaron deep learning?"

> "Por tres razones medidas, no por comodidad.
>
> El audio es telefónico de 8 kHz, y la literatura de anti-spoofing está
> entrenada en 16 kHz limpio donde los artefactos viven arriba de 4 kHz. La
> telefonía los borra: un modelo preentrenado llega peor de lo que promete.
>
> Viabilidad: esto corre en CPU, sin mandar audio del cliente a terceros. Para
> un banco eso no es preferencia, es requisito.
>
> Y explicabilidad: podemos enseñarles el peso exacto de cada medida."

### 4. "¿Cuánto tarda?"

> "El procesamiento son 171 milisegundos en CPU. Lo que ven en el servicio
> desplegado incluye subir 6 megas por internet y un servidor gratuito con una
> décima de núcleo. En una instancia normal de banco, esto corre por debajo de
> 200 milisegundos."

### 5. "Les salió 100%, ¿no están exagerando?"

**Díganlo antes de que pregunten.**

> "Acertamos 71 de 71, y no lo presentamos como titular. Son pocas llamadas:
> cero errores en 71 no significa que el error real sea cero. El número que
> reportamos es el 2.8% de la validación cruzada, sobre 282 llamadas y cinco
> particiones. Y su conjunto oculto tiene voces que no están en ninguno de
> nuestros grupos, así que esperamos degradación."

---

## Técnicas sobre el método

### "¿Funcionaría en tiempo real, durante la llamada?"

> "Las medidas de conversación se calculan de forma incremental: cada turno del
> agente da un dato nuevo. Con tres o cuatro intercambios ya hay señal, que son
> unos 30 segundos — antes de que el cliente dé cualquier dato sensible. No lo
> implementamos porque el reto pedía clasificar grabaciones, pero la
> arquitectura lo permite."

### "¿Qué pasa si la llamada es muy corta?"

> "Necesitamos al menos tres turnos de respuesta para que la medida de
> consistencia signifique algo. Por debajo de eso el sistema responde con
> confianza 0.5 en vez de adivinar. Lo probamos: con un audio de 0.2 segundos
> no truena, responde neutro."

### "¿Cómo eligieron el umbral?"

> "Con el EER de la validación cruzada dentro del grupo de entrenamiento, nunca
> mirando el de validación. Si lo ajustáramos con validación, ese número
> dejaría de significar algo."

### "¿Por qué 42 medidas y no más o menos?"

> "No optimizamos el número. Salieron de dos bloques: 23 de comportamiento y 19
> acústicas. Lo que sí probamos es quitarlas: sin las 7 de canal rinde igual, y
> solo con las 23 de comportamiento rinde casi igual. O sea que podríamos vivir
> con menos."

### "Expliquen una de las medidas"

Cualquiera, pero la mejor es el coeficiente de variación:

> "Tomamos los tiempos de respuesta y dividimos su desviación entre su
> promedio. Eso mide si responde *siempre igual*, sin importar si responde
> rápido o lento. Por eso un bot veloz seguiría delatándose: sería veloz pero
> parejo."

### "¿Cómo detectan cuándo habla cada quien?"

> "Detección por energía con umbral adaptativo. Adaptativo porque el nivel de
> voz varía 19 dB entre llamadas del dataset: un umbral fijo se pierde por
> completo a quienes hablan bajito. Medimos el silencio y la voz de cada
> llamada por separado y ponemos el corte entre los dos."

---

## Incómodas

### "¿Usaron IA para programarlo?"

> "Sí. Hicimos lluvia de ideas primero y usamos IA para aterrizarla y armar la
> arquitectura. El criterio fue nuestro: qué señal perseguir, qué medir, y
> cuándo desconfiar de un resultado que salía demasiado bien. Y podemos
> explicar cada pieza — de hecho reconstruimos el detector paso a paso para
> entenderlo."

**Dicho con naturalidad. Son ingenieros que construyen agentes de IA para
bancos; lo raro sería negarlo.**

### "Su cuarta medida más fuerte es el ruido de fondo. ¿No aprendieron el micrófono?"

> "Lo verificamos justo por eso. Esa medida sola acierta el 57%, apenas mejor
> que un volado. Y quitando las 7 medidas de canal el sistema rinde igual.
> Pesa dentro del conjunto, pero no está cargando la decisión."

### "71 llamadas de validación son muy pocas"

> "Estamos de acuerdo, por eso no reportamos ese resultado. El número que
> defendemos es el de la validación cruzada sobre 282."

### "¿Esto funciona con otros acentos o países?"

**No lo midieron. Sean honestos.**

> "No lo medimos. El dataset es español mexicano y no tenemos con qué probar
> otros. Lo que sí podemos decir es que la señal principal no depende del
> idioma: mide cuándo se habla, no qué se dice. Pero habría que comprobarlo."

### "¿Qué pasa si es una persona con el teléfono en altavoz, o con mala señal?"

> "El detector de voz se adapta al nivel de cada llamada, así que el volumen no
> nos afecta. Lo que sí podría afectarnos es que se corten las palabras y los
> turnos queden mal marcados. No lo medimos por separado, pero el dataset ya
> trae variedad de dispositivos y condiciones."

---

## Sobre el producto

### "¿Un banco podría desplegar esto?"

> "Corre en CPU, el modelo pesa 4 KB, no usa servicios externos y la voz del
> cliente nunca sale de la infraestructura. Además registramos cada consulta
> sin guardar el audio, por las mismas razones de privacidad."

### "¿Qué harían con una semana más?"

> "La señal semántica. Su agente pregunta por cosas que no existen: una persona
> dice 'no tengo eso', un modelo de lenguaje tiende a inventar. Es ortogonal a
> las dos que ya tenemos y sería la más difícil de evadir."

### "¿Y los falsos positivos? ¿Bloquear a un cliente real?"

> "Para un banco ese es el error caro. Nuestro umbral se puede mover: si
> prefieren no molestar nunca a un cliente real, se sube y se detectan menos
> bots pero no se bloquea a nadie por error. Devolvemos una confianza calibrada
> justamente para que esa decisión sea del banco, no nuestra."

---

## Si preguntan algo que no saben

No inventen. Cualquiera de estas sirve:

- "No lo medimos."
- "Eso no lo alcanzamos a probar."
- "No sé, pero lo mediría así: ..."

Y si alguien del equipo sabe la respuesta mejor, que la conteste esa persona.
Cuatro cabezas es una ventaja, úsenla.
