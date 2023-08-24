# MLOPs Project - Predicting House Prices (Keras - Artificial Neural Network)

## Introduction of the project

Our MLOps project proposes to address these challenges by merging data science and software engineering in a synergistic collaboration. The objective of the project is to carry out a proof of concept where the implementation of a machine learning model (Artificial Neural Network) will be sought, developing the different stages of MLOps, up to the use of Docker Compose for the deployment of the model.

## About the project

The overall goal of this project is to create a robust and reproducible MLOps workflow for developing, training, and deploying machine learning models. A neural network will be used as proof of concept due, and will be applied to the house price predict data set to predict the price of houses in King County, which includes Seattle.

This project covers the following topics:

1. **Key concepts of ML systems**  
The objective of this module is to give an introduction to MLOps, life cycle and architecture examples is also given.

2. **Basic concepts and tools for software development**  
This module focuses on introducing the principles of software development that will be used in MLOps. Consider the configuration of the environment, tools to use, and best practices, among other things.

3. **Development of ML models**  
This module consists of showing the development of an ML model from experimentation in notebooks, and subsequent code refactoring, to the generation of an API to serve the model.

4. **Deployment of ML models**  
The objective of this module is to show how a model is served as a web service to make predictions.

5. **Integration of concepts**  
This module integrates all the knowledge learned in the previous modules. A demo of Continuous Delivery is implemented.

### Baseline

This MLOps project focuses on demonstrating the implementation of a complete workflow ranging from data preparation to exposing a local web service to making predictions using a neural network. The chosen data set is home sales prices in King County, including Seattle. Includes houses for sale between May 2014 and May 2015.

The purpose is to establish a starting point or "baseline" that will serve as a reference to evaluate future improvements and not only more complex algorithms but more complex components and further deployments.

### Scope

This project is planned to cover the topics seen in the course syllabus, which was designed to include technical capacity levels 0, 1 and a small part of 2 of [Machine Learning operations maturity model - Azure Architecture Center | Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model).

In other words, knowledge is integrated regarding the learning of good software development practices and Dev Ops (Continuous Integration) applied to the deployment of ML models.

### Links to experiments like notebooks

You can find the predict house price experiments here:

* [1-exploring-data.ipynb](docs/notebooks/predicting_house_prices_keras_ann.ipynb)

## Setup

### Python version and packages to install

* Change the directory to the root folder.

* Create a virtual environment with Python 3.10+:

    ```bash
    python3.10 -m venv venv
    ```

* Activate the virtual environment

    ```bash
    source venv/bin/activate
    ```

* Install libraries
Run the following command to install the libraries/packages.

    ```bash
    pip install -r requirements.txt
    ```

## Model training from a house_price_prediction.py file

To train the neural network model, only run the following code:

```bash
python itesm_mlops_project/house_price_prediction.py
```

Output:

```bash
MAE:  105217.97457443322
MSE:  28882225899.6316
RMSE:  169947.7151939137
Variance Regression Score:  0.7944141570971985
Model saved in ./models/neural_network_model.pkl
```

## Execution of unit tests (Pytest)

### Test location

You can find the test location in the [test](tests) folder, and the following tests:

* Test `test_missing_indicator_transform`:  
Test the `transform` method of the MissingIndicator transformer.

* Test `test_missing_indicator_fit`:  
Test the `fit` method of the MissingIndicator transformer.

* Test `test_csv_file_existence`:  
Test case to check if the CSV file exists.

* Test `test_model_existence`:  
Test to validate the existence of a `.pkl` model file.

## Usage
### Individual Fastapi and Use Deployment
FastAPI is a modern web API development framework for Python that combines high performance with a simple declarative syntax. Designed to make it easy to create fast and efficient APIs, FastAPI has quickly gained popularity in the development community due to its speed, ease of use, and automatic generation of interactive documentation.

Uvicorn is a lightning-fast ASGI (Asynchronous Server Gateway Interface) server designed to run Python web applications asynchronously and efficiently. Being an ASGI implementation, Uvicorn takes advantage of the asynchronous nature of Python to handle multiple connections concurrently, enabling exceptionally high throughput and low latency in web applications.

* Run the next command to start the house predict API locally

    ```bash
    uvicorn itesm_mlops_project.api.main:app --reload
    ```

#### Checking endpoints

1. Access `http://127.0.0.1:8000/`, you will see a message like this `"house predict price is all ready to go!"`
2. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
![FastAPI Docs](docs/imgs/fast-api-docs.png)
3. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
            "bedrooms": 1,
            "bathrooms": 2,
            "sqft_living": 0,
            "sqft_lot": 0,
            "floors": 0,
            "waterfront": 0,
            "view": 0,
            "condition": 0,
            "grade": 0,
            "sqft_above": 0,
            "sqft_basement": 0,
            "yr_built": 0,
            "yr_renovated": 0,
            "lat": 0,
            "long": 0,
            "sqft_living15": 0,
            "sqft_lot15": 0,
            "month": 0,
            "year": 0
        }

        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [90957.64]"
        ```

    * **Prediction 2**  
        Request body

        ```bash
         {
            "bedrooms": 1,
            "bathrooms": 0,
            "sqft_living": 0,
            "sqft_lot": 0,
            "floors": 0,
            "waterfront": 0,
            "view": 0,
            "condition": 0,
            "grade": 0,
            "sqft_above": 0,
            "sqft_basement": 0,
            "yr_built": 0,
            "yr_renovated": 0,
            "lat": 0,
            "long": 0,
            "sqft_living15": 0,
            "sqft_lot15": 0,
            "month": 0,
            "year": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [73491.516]"
        ```


### Individual deployment of the API with Docker and usage

#### Build the image

* Ensure you are in the `itesm-mlops-project/` directory (root folder).
* Run the following code to build the image:

    ```bash
    docker build -t house-image ./itesm_mlops_project/app/
    ```

* Inspect the image created by running this command:

    ```bash
    docker images
    ```

    Output:

    ```bash
    REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
    house-image   latest    40045c4fb776   4 minutes ago   2.24GB
    ```

#### Run House Predict Price REST API

1. Run the next command to start the `house-image` image in a container.

    ```bash
    docker run -d --name house-container -p 8000:8000 house-image
    ```

2. Check the container running.

    ```bash
    docker ps -a
    ```

    Output:

    ```bash
    CONTAINER ID   IMAGE          COMMAND                  CREATED       STATUS       PORTS                    NAMES
    a5ed9e2af11c   463a87774a45   "uvicorn main:app --…"   4 hours ago   Up 4 hours   0.0.0.0:8000->8000/tcp   house-container
    ```

#### Checking endpoints for app

1. Access `http://127.0.0.1:8000/`, and you will see a message like this `"House predict price is all ready to go!"`
2. A file called `main.log` will be created automatically inside the container. We will inspect it below.
3. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
    ![FastAPI Docs](docs/imgs/fast-api-docs.png)

4. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
            "bedrooms": 1,
            "bathrooms": 1,
            "sqft_living": 1,
            "sqft_lot": 0,
            "floors": 0,
            "waterfront": 0,
            "view": 0,
            "condition": 0,
            "grade": 0,
            "sqft_above": 0,
            "sqft_basement": 0,
            "yr_built": 0,
            "yr_renovated": 0,
            "lat": 0,
            "long": 0,
            "sqft_living15": 0,
            "sqft_lot15": 0,
            "month": 0,
            "year": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```

        ![Prediction 1](docs/imgs/prediction-1.png)

    * **Prediction 2**  
        Request body

        ```bash
         {
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft_living": 2,
            "sqft_lot": 0,
            "floors": 0,
            "waterfront": 0,
            "view": 0,
            "condition": 0,
            "grade": 0,
            "sqft_above": 0,
            "sqft_basement": 0,
            "yr_built": 0,
            "yr_renovated": 0,
            "lat": 0,
            "long": 0,
            "sqft_living15": 0,
            "sqft_lot15": 0,
            "month": 0,
            "year": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [1]"
        ```

        ![Prediction 2](docs/imgs/prediction-2.png)
