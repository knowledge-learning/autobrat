# BRAT docker for SAT+R annotation schema

* Run `docker-compose build` to build the docker image.
* Run `docker-compose up` and navigate to [http://localhost:8080](http://localhost:8080).

## Notas de anotación

### Regla fundamental

- Al conectar conceptos compuestos, conectar las raíces de los respectivos sub-árboles. Los conexiones (salvo por relaciones semánticas) construyen un gran concepto del que se habla al referenciar la raíz. Cuidado con esto a la hora de contextualizar, pues en ocasiones habrá que anotar doble el mismo span de texto.

> **Ejemplo:** El dengue es una enfermedad que en estado avanzado es peligroso.
> - El dengue esté avanzado o no es una enfermedad.
> - Solo el dengue avanzado es peligroso.

### `same-as` vs `is-a` vs `has-property`

- Si la relación se cumple en ambos sentidos entonces usaremos `same-as`.
- En caso contrario, para el sentido correcto `X -> Y`, si `X` es subtipo o instancia de `Y` (implica que `Y` es un tipo o conjunto) entonces usaremos `is-a`.
- En otro caso, usaremos `has-property`.
  - Ejemplo: La alimentación es su principal fuente de energía.

### Oraciones de definición

- **`X` es un `Y` que hace `Z`:** se anota como `X -is-a-> Y` y se relaciona `X` con `Z`. Distinto de relacionar `Y` con `Z` y decir `X -same-as-> (Y,Z)`.

### `Predicate` vs contextualizar

- Se prefiere **contextualizar** por encima de **predicado** en caso de que la función semántica del predicado se ajuste a la del contextualizador y el objeto a contextualizar sea el dominio del predicado.

> Usar `predicate` (con domain) en los casos en los que parezca más una query (**ej**, "problemas médicos más comunes", "algunas personas").

> La idea es que hay una diferencia entre _"... el peso más bajo ..."_ (**<- predicate**) y _"... peso más bajo de lo normal ..."_ (**<- context**).

### Multiword vs `in-context`

- Se usa `in-context` solo cuando el contexto habla de un rasgo o valor que puede tener el concepto a contextualizar. Esto implica que el concepto a contextualizar debe poder existir sin el contexto (pero **ojo** que el concepto contextualizado tiene que ser equivalente al concepto completo, en el sentido de que si el concepto multiword tiene un significado distinto a la unión de sus partes, entonces se anota como multiword: **ej**, globulos blancos).

> Preguntarse si el concepto destino de `in-context` sería válido como `has-property`.

> Si la palabra completa está en wikipedia, entonces multiword es lo que se quiere.

### `Y -arg-> X` vs `Y -domain-> X`

- **Heurística:** usar `domain` si el _tipo_ o _clase_ del concepto resultante de formar el predicado (`Y`) coincide con el de `X`. No siempre es tan claro: **ej**, `tipo -domain-> cáncer` o `parte -domain-> cuerpo`.

En otro caso, `Y` tiene que ser un término sin relevancia por si solo, y en su lugar ser una especie de propiedad que puede ser medida o obtenida de `X`.

- "nivel de glucosa": `nivel -arg-> glucosa`.
- "problemas de salud" `problemas -arg-> salud`.
- "riesgo de cáncer" `riesgo -arg-> cáncer`.

### `Predicate -arg-> X` vs contextualizar

- Para contextualizar ambos deben ser conceptos relevantes del dominio. Lo contrario en el caso del predicado: al menos uno (o ambos) no es relevante para el dominio.

### `in-time` para duración

- Puede usarse para hablar de duración o momento de ocurrencia de otro concepto.
  > "... durante el embarazo": `-in-time-> embarazo`.

### `in-place` para contextualizador de `part-of`

- Cuando se quiere plantear un hecho sobre un concepto que es parte de otro, se puede usar `in-place` como contextualizador de `part-of` (en el sentido de que `part-of` es una relación semántica así que no se puede relacionar).

> **Ejemplo:** "... las arterias de su corazón ...": `arterias -in-place-> corazón`.

### `in-context` para pre-requisitos

- Para oraciones imperativas, como: _"Si X entonces haga Y"_, se impondrá `X` como contexto de `Y` (`Y -in-context-> X`).
  > **Ejemplo:** Si usted está preocupado acerca de la respiración de su hijo, llame a su proveedor de atención médica de inmediato.

### `action -> in-place` vs `action -> target`

- Preferiremos `action -> target` si la acción está actuando directamente sobre el concepto.

  > **Ejemplo:** "... dolor e hinchazón en las articulaciones ...". Usaremos `target`.

### `... para ...`

- Decidir entre `cause`, `entails`, `target` de un `action` o `arg` de un `predicate`.

    > **Ejemplo:** "... suficientes enzimas para descomponer los lípidos". En este caso se prefiere `suficientes -domain-> enzimas` `suficientes -arg-> descomponer`.


## Current annotation status

### Spanish news

| **Pack**      | **1st** | **2nd** | **Merged** | **Revised** | **Final** | **1st Annotator** | **2nd Annotator** |
|--|--|--|--|--|--|--|--|
|  1 |   |   |   |   |   |   |   |
|  2 |   |   |   |   |   |   |   |
|  3 |   |   |   |   |   |   |   |
|  4 |   |   |   |   |   |   |   |
|  5 |   |   |   |   |   |   |   |
|  6 |   |   |   |   |   |   |   |
|  7 |   |   |   |   |   |   |   |
|  8 |   |   |   |   |   |   |   |
|  9 |   |   |   |   |   |   |   |
| 10 |   |   |   |   |   |   |   |
| 11 |   |   |   |   |   |   |   |
| 12 |   |   |   |   |   |   |   |
| 13 |   |   |   |   |   |   |   |
| 14 |   |   |   |   |   |   |   |
| 15 |   |   |   |   |   |   |   |
| 16 |   |   |   |   |   |   |   |
| 17 |   |   |   |   |   |   |   |
| 18 |   |   |   |   |   |   |   |
| 19 |   |   |   |   |   |   |   |
| 20 |   |   |   |   |   |   |   |
| 21 |   |   |   |   |   |   |   |
| 22 |   |   |   |   |   |   |   |
| 23 |   |   |   |   |   |   |   |
| 24 |   |   |   |   |   |   |   |
| 25 |   |   |   |   |   |   |   |
| **Sentences** | | | |  |   |  |  |
