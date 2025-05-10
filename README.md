# Amazon Reviews Sentiment Analysis

## Descripción del Proyecto
Este proyecto implementa un flujo completo de recolección y análisis de datos para clasificar el **sentimiento** (positivo, neutro o negativo) de reseñas de productos en **Amazon.es**, incluyendo:

1. Recolección y filtrado de datos  
2. Análisis exploratorio de los datos (EDA)  
3. Limpieza y preprocesamiento de texto  
4. Visualizaciones clave  
5. Hallazgos preliminares y conclusiones  

## 1. Recolección y Filtrado
Descargamos el **Multilingual Amazon Reviews Corpus** y unimos los splits `train`, `validation` y `test`, quedándonos solo con las reseñas en **español**.

```python
import kagglehub, pandas as pd, os

root = kagglehub.dataset_download("mexwell/amazon-reviews-multi")
load_es = lambda split: (
    pd.read_csv(os.path.join(root, f"{split}.csv"), encoding="latin-1")
      .query('language=="es"')
      .drop(columns="language")
)
df = pd.concat([load_es(s) for s in ["train","validation","test"]], ignore_index=True)
df.to_csv("amazon_reviews_es.csv", index=False)
```

- **Total registros:** 210 000 reseñas en español.  
- **Calidad inicial:** sin valores nulos ni duplicados (ver `df.info()`).

## 2. Inspección Inicial
Verificamos la calidad de datos, duplicados y nulos, y obtenemos estadísticas descriptivas.

```python
df.info()
print("Duplicados:", df.duplicated().sum())
print("Nulos por columna:
", df.isna().sum())
```

## 3. Limpieza y Preprocesamiento de Texto
Usamos **spaCy** para normalizar, tokenizar y lematizar:

```python
import spacy, re, unicodedata

nlp = spacy.load("es_core_news_sm", disable=["parser","ner"])
STOP_ES = nlp.Defaults.stop_words

def preprocess(text):
    txt = unicodedata.normalize("NFD", text.lower()) \
                     .encode("ascii","ignore").decode()
    txt = re.sub(r"http\S+|www\S+|\d+|[^a-zñáéíóúü\s]", " ", txt)
    return txt

cleaned = []
for doc in nlp.pipe(df.review_body.map(preprocess),
                    batch_size=2000, n_process=4):
    tokens = [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOP_ES]
    cleaned.append(" ".join(tokens))

df["clean_body"] = cleaned
df["length"] = df.clean_body.str.split().str.len()
df.to_csv("amazon_reviews_es_clean_spacy.csv", index=False)
```

## 4. Visualizaciones Exploratorias
- **Distribución de calificaciones**  
  ```python
  df.stars.value_counts().sort_index().plot.bar()
  ```
- **Longitud de reseña vs. calificación**  
  ```python
  df.boxplot(column="length", by="stars")
  ```
- **Top-10 categorías de producto**  
  ```python
  df.product_category.value_counts().head(10).plot.bar()
  ```
- **Top-30 lemas**  
  ```python
  from collections import Counter
  Counter(" ".join(df.clean_body).split()).most_common(30)
  ```
- **Nubes de palabras positivas vs negativas**  
  ```python
  from wordcloud import WordCloud
  pos = " ".join(df.query("stars>=4").clean_body)
  neg = " ".join(df.query("stars<=2").clean_body)
  WordCloud().generate(pos)
  WordCloud().generate(neg)
  ```

## 5. Hallazgos Clave
- **Composición del Dataset**  
  - 210 000 reseñas limpias, sin nulos ni duplicados.  
  - Distribución uniforme de calificaciones (~ 42 000 reseñas por estrella).  
- **Características del Texto**  
  - Longitud media: 10.6 palabras. RIC: 5–13 (colas hasta ~ 200 palabras).  
  - Reseñas de 1–2 ★ ligeramente más extensas.  
- **Categorías de Producto**  
  - 30 categorías; las 10 principales concentran ~ 67 % de las reseñas.  
- **Análisis Léxico**  
  - Lemas más comunes: calidad, producto, precio, funcionar, comprar.  
  - Nube positiva: “perfecto”, “bonito”, “encantar”.  
  - Nube negativa: “funcionar”, “devolver”, “llegar”.

## 6. Próximos Pasos
- Experimentar con modelos multinivel (Regresión Logística, Naive Bayes, Transformers).  
- Incorporar variables adicionales (n-gramas, embeddings, categoría).  
- Aplicar técnicas de explicabilidad (LIME/SHAP) sobre lemas dominantes.

## Citación del Dataset
Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). _The Multilingual Amazon Reviews Corpus_. Proceedings of EMNLP 2020.  
Kaggle: https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi  
