# DeepDetect 2026

**Asignatura:** IA Generativa para la Gestión Industrial  
**Módulo:** II, clasificación de imágenes  
**Integrantes:** Catalina Cornejo, Hellen Quiñones y Gustavo Sánchez  
**Repositorio GitHub:** [DeepDetect_2026_Proyecto](https://github.com/hellenelizabethqm-netizen/DeepDetect_2026_Proyecto/tree/main)  
**Aplicación web pública:** (https://deepdetect2026-gnu6pnqzy6yztfrxfdfjkl.streamlit.app/)

## 1. Descripción y propósito

IA Detect 2026 es una aplicación educativa que apoya la identificación de imágenes faciales posiblemente generadas o manipuladas mediante inteligencia artificial. El proyecto responde a los riesgos de suplantación visual y desinformación mediante un clasificador binario que distingue entre las categorías `fake` y `real`, presenta la puntuación del modelo y ofrece recomendaciones para revisar el origen del contenido.

Su finalidad es facilitar una primera evaluación de las imágenes, sin reemplazar la verificación humana ni presentar las predicciones como pruebas definitivas de autenticidad.

## 2. Origen y características del dataset

Se utiliza [Deep Detect Dataset 2026](https://www.kaggle.com/datasets/ucimachinelearning/deep-detect-dataset-2026?select=DeepDetect), publicado en Kaggle mediante la cuenta `ucimachinelearning` bajo la licencia **CC0: Public Domain**. De acuerdo con su descripción, es un conjunto curado para investigar la detección automatizada de *deepfakes* y diferenciar fotografías faciales auténticas de imágenes alteradas digitalmente o generadas de forma sintética.

La categoría `real` reúne imágenes faciales etiquetadas como auténticas y sin manipulación, mientras que `fake` incluye imágenes faciales manipuladas o sintéticas. La ficha compartida no detalla la procedencia individual de todas las fotografías ni de sus métodos de generación, aspecto que constituye una limitación para valorar la representatividad del conjunto.

Durante la auditoría del notebook se examinaron **21.776 imágenes**, todas legibles, y se eliminaron **199 duplicados exactos**, obteniendo **21.577 imágenes** para el análisis. También se identificaron coincidencias perceptuales mediante pHash, consideradas en la separación de los datos para reducir filtraciones entre conjuntos.

| Categoría | Imágenes válidas | Proporción |
| :--- | ---: | ---: |
| `fake` | 10.399 | 48,19 % |
| `real` | 11.178 | 51,81 % |
| **Total** | **21.577** | **100 %** |

Las clases presentan una distribución relativamente equilibrada, por lo que no se observa un desbalance importante. No obstante, la calidad de las etiquetas y las diferencias de origen, iluminación, compresión o manipulación pueden influir en el aprendizaje y la generalización de los modelos.

## 3. Preparación y división de los datos

Se realizó una separación estratificada con semilla **2026**, utilizando como referencia un 70 % de las imágenes para entrenamiento, un 15 % para validación y un 15 % para prueba. Las imágenes con el mismo hash perceptual se mantuvieron dentro de un único conjunto, de modo que las proporciones finales presentan pequeñas variaciones respecto de la distribución propuesta.

| Conjunto | `fake` | `real` | Total | Proporción |
| :--- | ---: | ---: | ---: | ---: |
| Entrenamiento | 7.278 | 7.820 | 15.098 | 69,97 % |
| Validación | 1.561 | 1.680 | 3.241 | 15,02 % |
| Prueba | 1.560 | 1.678 | 3.238 | 15,01 % |

El conjunto de entrenamiento se utilizó para ajustar los parámetros de los modelos, mientras que validación permitió seleccionar la arquitectura y el umbral de decisión. Por su parte, prueba se reservó para la evaluación final, sin intervenir en la selección, favoreciendo una comparación más objetiva y reproducible.

## 4. Modelos y criterios de selección

Se compararon **EfficientNet-B0** y **ResNet-50**, dos arquitecturas preentrenadas implementadas con `timm` y adaptadas mediante *transfer learning*. EfficientNet-B0 se consideró por su menor costo computacional, mientras que ResNet-50 permitió contrastar el desempeño con una red residual ampliamente utilizada en clasificación de imágenes.

Ambos modelos completaron seis épocas de entrenamiento de la nueva capa clasificadora y tres épocas de ajuste fino, utilizando `BCEWithLogitsLoss` y AdamW. Las etiquetas se codificaron como `fake = 0` y `real = 1`, de manera que la salida sigmoide corresponde a `P(real)` y la puntuación para la clase sintética se obtiene como `P(fake) = 1 − P(real)`.

La selección se realizó mediante el **F2 de la clase `fake` en validación**, métrica que asigna mayor importancia al *recall* y permite priorizar la identificación de imágenes sintéticas. ResNet-50 alcanzó un F2 de **99,14 %**, frente al **98,67 %** de EfficientNet-B0; por ello, se seleccionó **ResNet-50** con un umbral de **0,76 para `P(real)`**, definido antes de utilizar el conjunto de prueba.

## 5. Evaluación y resultados

Ambos modelos se evaluaron sobre las mismas **3.238 imágenes de prueba**, utilizando métricas equivalentes para asegurar una comparación consistente.

| Métrica en prueba | EfficientNet-B0 | ResNet-50 |
| :--- | ---: | ---: |
| Accuracy | 97,87 % | **98,58 %** |
| Balanced accuracy | 97,91 % | **98,62 %** |
| Precision de `fake` | 96,74 % | **97,43 %** |
| Recall de `fake` | 98,91 % | **99,68 %** |
| F1 de `fake` | 97,81 % | **98,54 %** |
| F2 de `fake` | 98,47 % | **99,22 %** |
| Imágenes `fake` clasificadas como `real` | 17 | **5** |
| Errores totales | 69 | **46** |

**ResNet-50 obtuvo el mejor desempeño general**, con 1.555 de las 1.560 imágenes `fake` correctamente identificadas. Solo cinco imágenes sintéticas fueron clasificadas como reales, error especialmente relevante porque podría generar confianza indebida en contenido manipulado; en sentido contrario, 41 imágenes reales se clasificaron como `fake`, lo que demuestra la necesidad de interpretar las alertas con prudencia.

Las curvas de aprendizaje y las matrices de confusión incluidas en el notebook respaldan esta comparación. Asimismo, el análisis visual de algunos errores sugiere dificultades ante retratos muy uniformes, imágenes comprimidas o rostros sintéticos de apariencia convincente, sin que estas observaciones permitan establecer por sí solas las causas de cada predicción incorrecta.

## 6. Aplicación web e integración de IA generativa

La aplicación **IA Detect 2026** se desarrolló con **Streamlit** para permitir la carga de imágenes y mostrar la clase estimada, las puntuaciones de `fake` y `real`, el desempeño global del modelo y advertencias cuando una predicción resulta cercana al umbral seleccionado. Para las nuevas imágenes, se utiliza ResNet-50 con los parámetros y el preprocesamiento exportados desde el notebook.

La interfaz incorpora un botón para solicitar recomendaciones a **Google Gemini** a partir de la clase y las puntuaciones calculadas. El modelo de lenguaje recibe estos antecedentes como texto, sin recibir la imagen, y propone acciones de verificación como revisar la fuente original, contrastar el contexto y utilizar herramientas de búsqueda inversa, evitando presentar una clasificación automática como certeza.

La conexión con Gemini se probó satisfactoriamente desde Google Colab y también se inició una versión temporal de Streamlit en ese entorno. Para el despliegue definitivo será necesario verificar el funcionamiento completo de la aplicación desde su enlace público y mantener `GEMINI_API_KEY` en los secretos de la plataforma, sin exponerla en GitHub.

**Enlace público de la aplicación:** [Pendiente de incorporar y verificar].

## 7. Ejecución y reproducibilidad

El entrenamiento se realizó en Google Colab mediante la descarga del dataset con `kagglehub`, la auditoría de archivos, la partición con semilla 2026 y la comparación de los dos modelos. Al finalizar, se exportaron los pesos de **ResNet-50**, su configuración y las métricas para emplearlos en la aplicación.

Para iniciar Streamlit en un entorno local con los archivos del modelo disponibles, se instalan las dependencias y se ejecuta:

```bash
pip install -r requirements.txt
streamlit run app.py
```

La clave de Gemini se configura como variable de entorno o secreto de Streamlit. La publicación debe comprobarse desde otro navegador o dispositivo, verificando la carga de imágenes, los resultados de clasificación, las métricas, las advertencias y las recomendaciones generativas.

## 8. Limitaciones y uso responsable

El desempeño obtenido describe el comportamiento del modelo frente a la distribución de imágenes contenida en Deep Detect Dataset 2026, pero no garantiza resultados equivalentes ante generadores distintos, fotografías de otras procedencias o nuevas formas de manipulación. Además, el redimensionamiento y la compresión pueden modificar señales visuales relevantes, mientras que el agrupamiento por pHash no identifica necesariamente todas las imágenes relacionadas por una misma escena o fuente.

Las puntuaciones mostradas no son una certificación de autenticidad ni probabilidades calibradas de verdad factual. Por esta razón, el sistema mantiene un enfoque educativo, recomienda la revisión humana y no debe utilizarse de manera aislada para formular acusaciones o fundamentar decisiones legales, económicas o disciplinarias.

## 9. Uso de inteligencia artificial en el desarrollo

Se utilizó **ChatGPT (OpenAI)** como apoyo en la estructuración del notebook, la generación y revisión inicial del código, la documentación y el diseño de instrucciones para el modelo de lenguaje. Asimismo, se integró **Google Gemini** para elaborar recomendaciones orientadas a verificar la procedencia de las imágenes; las decisiones metodológicas, la ejecución del entrenamiento y la interpretación de los resultados corresponden al trabajo del equipo.


## Referencias

Kaggle. *Deep Detect Dataset 2026*. https://www.kaggle.com/datasets/ucimachinelearning/deep-detect-dataset-2026

Hugging Face. *timm documentation*. https://huggingface.co/docs/timm/index

PyTorch. *Transfer learning for computer vision*. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Scikit-learn. *Classification metrics*. https://scikit-learn.org/stable/modules/model_evaluation.html

Material del módulo: *Pauta 1, Clasificación binaria de imágenes con una CNN* y *Pauta 2, Alzheimer con tres modelos preentrenados*, proporcionados por el docente.
