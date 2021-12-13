# covid19-prediction

Models to predict covid-19 cases in India and USA

Report title: *Covid-19 Forecasting with Vaccinations as a factor: the case of India and USA*
## Team Details
**Team No: 60**     
**Team Name: Vanadium**     
Team Members:
- Vishruth Veerendranath (PES1UG19CS577)
- Vibha Masti (PES1UG19CS565)
- Harshith Mohan Kumar (PES1UG19CS276)

## Data sources

The data were sourced from the following sources:

1. Daily state-wise COVID-19 cases for India: [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19/tree/ef15d99458d44aa9bc03c0726c609643e6f90d3b)

2. Daily state-wise COVID-19 vaccinations for India: [COVID-19 India API - cowin_vaccine_data_statewise](https://data.covid19india.org)

3. Daily state-wise COVID-19 cases for USA: todo

4. Daily state-wise COVID-19 vaccination for USA: [Data on COVID-19 (coronavirus) vaccinations by Our World in Data](https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations)

## Data cleaning

The data were cleaned in our [eda repository](https://github.com/Data-Analytics-Project-2021/eda) and are stored under `cleaned_datasets/`.

The raw sourced data are store in `raw_datasets/`.

## Docker

### Development

To develop/maintain code use the following steps to setup your environment.

#### Docker Container

1. To build the docker dev image run the following command

```
docker-compose up
```

2. Next use the following command to start up the dev docker container.

```
docker run --gpus all -it --rm -p 8888:8888 -v $PWD:/covid19-prediction covid19-prediction_dev
```

#### Jupyter Notebook

Once the container is up and running use the following code to launch jupyter notebooks.

```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

#### Debugging Errors

1. error checking context: 'can't stat 'error checking context: 'can't stat '...error checking context: 'can't stat'

Solution
```
ls -a
sudo rm -r .Trash-0/
docker-compose up
```