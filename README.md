# Short Term Load Forecasting using feature engineering 

This repository provides the code for the paper "Short-Term Load Forecasting using Bi-directional Sequential Models and Feature Engineering for Small Datasets" available at <a href=https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9467267> paper </a>.

<img src="https://github.com/manastahir/Short-Term-Load-Forecasting/blob/master/alt/Architechure.png" width="800" height="400">

<hr/>

## Introduction 
In the paper tests were performed on five different datasets: Smart Grid Smart City (SGSC), The Almanac of Minutely Power dataset (AMPD), Réseau de Transport d’Électricité (RTE),
The Electric Reliability Council of Texas (ERCOT), Pakistan Residential Electricity Consumption (PRECON).In this repository we provide the code for the derived models and 
derived feature generation. The code provided is generic and needs to be slightly modified to fit the format of each dataset.  

## Dependencies
Make sure you have Python>=3.6 installed on your machine.
```shell
pip install -q requirements.txt
```
## Training
Training and Testing and Graph Generation funcations arfe provided in /src/utils.py, which are imported in <b>Experiments.ipynb</b>. The experiments notebook contains code for 
agressive training and testing models on a particular dataset.

Set the parameters to select differnet model types, evaluation and loss functions and features. 

<b>Example:</b> 

```shell
experiment_type = "derived"
loss_function = "mape"
windows = [12]
model_names = ['LSTM']
```
Runs experiment using LSTM model with MAPE loss fucntion and 12 timesteps and derived features.

## Datasets

```shell
SGSC
@manual{smart_grid,
    title  = "Smart-Grid Smart-City Customer Trial Data",
    author = "",
    note   = "\url{https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef}",
    year   = "2019 (Accessed online:10.09.2019)"
}
```

```shell
RTE
@manual{rte-data,
    title  = "RTE, Grid data.",
    author = "",
    note   = "\url{https://data.rte-france.com/}",
    year   = "2019 (Accessed online:27.08.2019)"
}
```

```shell
ERCOT
@manual{ercot-data,
    title  = "ERCOT, Grid data.",
    author = "",
    note   = "\url{https://ercot.com/}",
    year   = "2019 (Accessed online:27.08.2019)"
} 
```

```shell
AMPD
@inproceedings{ampd,
    author={S. {Makonin} and F. {Popowich} and L. {Bartram} and B. {Gill} and I. V. {Bajić}},
    booktitle={2013 IEEE Electrical Power Energy Conference},
    title={{AMPds}: A public dataset for load disaggregation and eco-feedback research},
    year={2013},
    volume={},
    number={},
    pages={1-6},
    keywords={computerised monitoring;decision making;domestic appliances;energy conservation;home automation;load forecasting;natural gas technology;power system measurement;smart meters;eco-feedback research;home-based intelligent energy conservation system;home appliances;intelligent feedback;intelligent decision making;nonintrusive load monitoring;NILM;real power reading;Almanac of Minutely power dataset;submeter;AMPds;natural gas;water consumption;load disaggregation algorithm;Current measurement;Power measurement;Home appliances;Voltage measurement;Natural gas;Data acquisition;Power Meter;Current;Dataset;Load Disaggre-gation;Eco-Feedback;Single-Measurement;Maximum a Posteriori (MAP);Energy Conservation},
    doi={10.1109/EPEC.2013.6802949},
    ISSN={},
    month={Aug}
}
```

```shell
PRECON
@inproceedings{Nadeem:2019:PPR:3307772.3328317,
    author = {Nadeem, Ahmad and Arshad, Naveed},
    title = {{PRECON}: {Pakistan} Residential Electricity Consumption Dataset},
    booktitle = {Proceedings of the Tenth ACM International Conference on Future Energy Systems},
    series = {e-Energy '19},
    year = {2019},
    isbn = {978-1-4503-6671-7},
    url = {http://doi.acm.org/10.1145/3307772.3328317},
    doi = {10.1145/3307772.3328317},
    publisher = {ACM},
    keywords = {Consumption, Dataset, Electricity, PRECON},
}
```


## Citation
```shell
@inproceedings{IEEE Access,
    author = {Abdul Wahab, Muhammad Anas Tahir, Naveed Iqbal, Adnan Ul-Hasan, Faisal Shafiat, Syed Muhammad Raza Kazmi},
    title = {A Novel Technique for Short-Term LoadForecasting using Sequential Modelsand Feature Engineering},
    year = {2021},
    keywords = {Load Forecasting, Smart Grids, Deep Learning, Feature Engineering, Sequential Models},
}
```
