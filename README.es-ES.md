# Una primera mirada a la clasificación de Conventional Commits

<div align="center">

[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/0x404/ccs_dataset)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green)](https://huggingface.co/0x404/ccs-code-llama-7b)
[![Figshare](https://img.shields.io/badge/Figshare-005C47)](https://doi.org/10.6084/m9.figshare.26507083)

</div>

[Conventional Commits🪄](https://www. conventionalcommits.org/en/v1.0.0/), como una especificación para añadir significado legible tanto para humanos como para máquinas a los mensajes de commit, está ganando popularidad cada vez más entre los proyectos de código abierto y los desarrolladores. Llevamos a cabo un estudio preliminar de CCS, que abarca su estado de aplicación y los desafíos que encuentran los desarrolladores al utilizarlo. Observamos una creciente popularidad de CCS, sin embargo, los desarrolladores clasifican erróneamente los commits en tipos de CCS incorrectos, lo cual es atribuible a la ausencia de una lista de definiciones clara y distinta para cada tipo. Para solucionar esto, hemos desarrollado una lista de definiciones más precisa y con menos solapamientos, basada en las prácticas de la industria y en una revisión bibliográfica. Para ayudar a los desarrolladores a clasificar los conventional commits, proponemos un enfoque para la clasificación automatizada de conventional commits.

Este repositorio contiene todos los datos y el código que utilizamos en el estudio.

## Reproducción

### Usando Hugging Face (Recomendado)

Hemos subido el conjunto de datos y los parámetros del modelo al hub de Hugging Face, facilitando enormemente la replicación de nuestros resultados. Primero, asegúrese de haber instalado el entorno necesario:

```shell
pip3 install transformers[torch] datasets scikit-learn sentencepiece protobuf
```

Luego, puede utilizar `transformers` y `datasets` para cargar nuestro modelo y conjunto de datos, y probarlo en el conjunto de prueba:

```python
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

test_dataset = load_dataset("0x404/ccs_dataset", split="test")
pipe = pipeline("text-generation", model="0x404/ccs-code-llama-7b", device_map="auto")

outputs = pipe(test_dataset["input_prompt"], max_new_tokens=10, pad_token_id=pipe.tokenizer.eos_token_id)
predicted_labels = [output[0]["generated_text"].split()[-1] for output in outputs]

accuracy = accuracy_score(test_dataset["annotated_type"], predicted_labels)
f1 = f1_score(test_dataset["annotated_type"], predicted_labels, average="macro")

print("Accuracy:", accuracy)
print("F1 Score (Macro):", f1)
```

### Usando el código desde cero

El conjunto de datos está disponible en Hugging Face, lo que facilita mucho su acceso:

```python
from datasets import load_dataset
ccs_dataset = load_dataset("0x404/ccs_dataset")
```

Adicionalmente, en el directorio `Dataset` de este repositorio, proporcionamos dos conjuntos de datos: uno que contiene 88,704 commits en formato Conventional Commits extraídos de 116 repositorios, y otro conjunto de datos de 2,000 commits que fueron muestreados y anotados manualmente. El conjunto de datos anotado manualmente se utiliza para el entrenamiento, validación y prueba del modelo. Para obtener información detallada sobre los conjuntos de datos, consulte el README.md en el directorio `Dataset`.

Para ejecutar nuestro código desde cero, debe instalar los entornos necesarios. Proporcionamos la información de versión de todos los entornos dependientes en `requirements.txt`, que se puede instalar con:

```shell
pip3 install -r requirements.txt
```

- Para replicar los resultados experimentales de ChatGPT4, acceda al código en el directorio `RQ3/ChatGPT`. Incluye un archivo `test.csv` que es nuestro conjunto de datos de prueba. Inserte su clave de OpenAI (con acceso a ChatGPT4) en el código y luego ejecútelo.
- Para replicar los resultados experimentales de BERT, navegue al directorio `RQ3/BERT` y use el comando `python3 main.py --train` para el entrenamiento, y `python3 main.py --test <checkpointpath>` para las pruebas, donde `<checkpointpath>` es la ubicación del checkpoint de parámetros guardado.
- Para replicar los experimentos de Llama2 y CodeLlama, encuentre el código en los directorios `RQ3/Llama2` y `RQ3/CodeLlama` respectivamente. El entrenamiento se ejecuta con `python3 train.py`, y las pruebas con `python3 infer.py --checkpoint <checkpointpath>`, donde `<checkpointpath>` es la ubicación del checkpoint, por defecto en el directorio `llama-output` dentro del directorio actual.

*Nota: Para replicar Llama2 y CodeLlama, se requieren aproximadamente dos GPUs de 24GB de memoria. Si el hardware adecuado no está disponible, para facilitar la replicación, proporcionamos los parámetros LORA entrenados en el directorio `CheckPoints`. Si desea utilizar directamente los parámetros preentrenados de CodeLlama, ejecute el siguiente comando:

```shell
cd RQ3/CodeLlama
python3 infer.py --checkpoint ../../CheckPoints/CodeLlama-checkpoints
```

## Cómo usar

Nuestro modelo acepta básicamente dos entradas: el mensaje del commit y el git diff correspondiente. Para clasificar el commit deseado, debe seguir un formato de prompt específico. Proporcionamos el siguiente fragmento de código:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="0x404/ccs-code-llama-7b", device_map="auto")
tokenizer = pipe.tokenizer


def prepare_prompt(commit_message: str, git_diff: str, context_window: int = 1024):
    prompt_head = "<s>[INST] <<SYS>>\nYou are a commit classifier based on commit message and code diff.Please classify the given commit into one of the ten categories: docs, perf, style, refactor, feat, fix, test, ci, build, and chore. The definitions of each category are as follows:\n**feat**: Code changes aim to introduce new features to the codebase, encompassing both internal and user-oriented features.\n**fix**: Code changes aim to fix bugs and faults within the codebase.\n**perf**: Code changes aim to improve performance, such as enhancing execution speed or reducing memory consumption.\n**style**: Code changes aim to improve readability without affecting the meaning of the code. This type encompasses aspects like variable naming, indentation, and addressing linting or code analysis warnings.\n**refactor**: Code changes aim to restructure the program without changing its behavior, aiming to improve maintainability. To avoid confusion and overlap, we propose the constraint that this category does not include changes classified as ``perf'' or ``style''. Examples include enhancing modularity, refining exception handling, improving scalability, conducting code cleanup, and removing deprecated code.\n**docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.\n**test**: Code changes that modify test files, including the addition or updating of tests.\n**ci**: Code changes to CI (Continuous Integration) configuration files and scripts, such as configuring or updating CI/CD scripts, e.g., ``.travis.yml'' and ``.github/workflows''.\n**build**: Code changes affecting the build system (e.g., Maven, Gradle, Cargo). Change examples include updating dependencies, configuring build configurations, and adding scripts.\n**chore**: Code changes for other miscellaneous tasks that do not neatly fit into any of the above categories.\n<</SYS>>\n\n"
    prompt_head_encoded = tokenizer.encode(prompt_head, add_special_tokens=False)

    prompt_message = f"- given commit message:\n{commit_message}\n"
    prompt_message_encoded = tokenizer.encode(prompt_message, max_length=64, truncation=True, add_special_tokens=False)

    prompt_diff = f"- given commit diff: \n{git_diff}\n"
    remaining_length = (context_window - len(prompt_head_encoded) - len(prompt_message_encoded) - 6)
    prompt_diff_encoded = tokenizer.encode(prompt_diff, max_length=remaining_length, truncation=True, add_special_tokens=False)

    prompt_end = tokenizer.encode(" [/INST]", add_special_tokens=False)
    return tokenizer.decode(prompt_head_encoded + prompt_message_encoded + prompt_diff_encoded + prompt_end)


def classify_commit(commit_message: str, git_diff: str, context_window: int = 1024):
    prompt = prepare_prompt(commit_message, git_diff, context_window)
    result = pipe(prompt, max_new_tokens=10, pad_token_id=pipe.tokenizer.eos_token_id)
    label = result[0]["generated_text"].split()[-1]
    return label

```

Aquí, puede utilizar la función `classify_commit` para clasificar su commit ingresando el mensaje del commit y el git diff. El `context_window` controla el tamaño de todo el prompt, configurado en 1024 por defecto pero ajustable a un valor mayor como 2048 para incluir más git diff en un solo prompt. Aquí hay un ejemplo de su uso:

```python
import requests
from github import Github

def fetch_message_and_diff(repo_name, commit_sha):
    g = Github()
    try:
        repo = g.get_repo(repo_name)
        commit = repo.get_commit(commit_sha)
        if commit.parents:
            parent_sha = commit.parents[0].sha
            diff_url = repo.compare(parent_sha, commit_sha).diff_url
            return commit.commit.message, requests.get(diff_url).text
        else:
            raise ValueError("No parent found for this commit, unable to retrieve diff.")
    except Exception as e:
        raise RuntimeError(f"Error retrieving commit information: {e}")

message, diff = fetch_message_and_diff("pytorch/pytorch", "9856bc50a251ac054debfdbbb5ed29fc4f6aeb39")
print(classify_commit(message, diff))
```

En esta configuración, hemos definido una función `fetch_message_and_diff` que obtiene el mensaje del commit y el diff para cualquier SHA especificado de un repositorio de GitHub, permitiendo que nuestro modelo clasifique el commit en consecuencia.

## Rendimiento de tipos específicos de CCS

Esta tabla es la tabla completa proporcionada en nuestra RQ3, que incluye la precisión, el recall y la puntuación f1 para cada uno de los diez tipos específicos de CCS, con la puntuación más alta resaltada en **negrita**.

| Metrics            | BERT       | ChatGPT4   | Llama2     | Our Approach |
|:-------------------|:-----------:|:-----------:|:-----------:|:------------:|
| build_precision    | 0.6304     | **0.8286** | 0.6905     | 0.7442      |
| build_recall       | 0.725      | 0.725      | 0.725      | **0.8**     |
| build_f1           | 0.6744     | **0.7733** | 0.7073     | 0.7711      |
| ci_precision       | **0.8718** | 0.8571     | 0.8409     | 0.8605      |
| ci_recall          | 0.85       | 0.9        | **0.925**  | 0.925       |
| ci_f1              | 0.8608     | 0.878      | 0.881      | **0.8916**  |
| docs_precision     | 0.8095     | **0.8974** | 0.7451     | 0.8372      |
| docs_recall        | 0.85       | 0.875      | **0.95**   | 0.9         |
| docs_f1            | 0.8293     | **0.8861** | 0.8352     | 0.8675      |
| perf_precision     | 0.3939     | **0.9545** | 0.875      | 0.8378      |
| perf_recall        | 0.65       | 0.525      | 0.7        | **0.775**   |
| perf_f1            | 0.4906     | 0.6774     | 0.7778     | **0.8052**  |
| chore_precision    | 0.3846     | 0.6957     | 0.6129     | **0.7391**  |
| chore_recall       | **0.5**    | 0.4        | 0.475      | 0.425       |
| chore_f1           | 0.4348     | 0.5079     | 0.5352     | **0.5397**  |
| test_precision     | 0.6923     | 0.9        | 0.8889     | **0.9459**  |
| test_recall        | 0.675      | 0.675      | 0.8        | **0.875**   |
| test_f1            | 0.6835     | 0.7714     | 0.8421     | **0.9091**  |
| fix_precision      | 0.4167     | 0.4808     | **0.6829** | 0.6667      |
| fix_recall         | 0.25       | 0.625      | **0.7**    | 0.7         |
| fix_f1             | 0.3125     | 0.5435     | **0.6914** | 0.6829      |
| refactor_precision | 0.2414     | 0.4545     | **0.5814** | 0.5085      |
| refactor_recall    | 0.175      | 0.625      | 0.625      | **0.75**    |
| refactor_f1        | 0.2029     | 0.5263     | 0.6024     | **0.6061**  |
| style_precision    | 0.5333     | **0.8966** | 0.8049     | 0.7805      |
| style_recall       | 0.2        | 0.65       | **0.825**  | 0.8         |
| style_f1           | 0.2909     | 0.7536     | **0.8148** | 0.7901      |
| feat_precision     | 0.4583     | 0.5205     | 0.8205     | **0.875**   |
| feat_recall        | 0.55       | **0.95**   | 0.8        | 0.7         |
| feat_f1            | 0.5        | 0.6726     | **0.8101** | 0.7778      |
| macro_precision    | 0.5432     | 0.7486     | 0.7543     | **0.7795**  |
| macro_recall       | 0.5425     | 0.695      | 0.7525     | **0.765**   |
| macro_f1           | 0.528      | 0.699      | 0.7497     | **0.7641**  |
| accuracy           | 0.5425     | 0.695      | 0.7525     | **0.765**   |

## Estructura de archivos
```
.
├── CheckPoints: Contiene checkpoints de los parámetros de nuestros modelos ajustados
├── Dataset: Conjuntos de datos creados y utilizados en nuestra investigación
├── README.md: Descripción de este repositorio
├── requirements.txt: Dependencias del entorno
├── RQ1: Datos y código utilizados en RQ1
├── RQ2: Análisis de los desafíos de los desarrolladores en RQ2
└── RQ3: Código utilizado para el entrenamiento de modelos en RQ3
```

## Cítanos

Si utiliza este repositorio en su investigación, por favor cítenos utilizando la siguiente entrada BibTeX:

```bibtex
@inproceedings{zeng2025conventional,
  title={A First Look at Conventional Commits Classification},
  author={Zeng, Qunhong and Zhang, Yuxia and Qiu, Zhiqing and Liu, Hui},
  booktitle={Proceedings of the IEEE/ACM 47th International Conference on Software Engineering},
  year={2025}
}
```
