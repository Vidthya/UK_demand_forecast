### UK Electricity Demand Forecast



A learning project on public data. I forecast UK national electricity demand a day ahead, then asked what happens when the model is trained once and left alone, and whether retraining it helps. 



##### Main finding

A model trained on 2012-2018 and left unchanged beat simple baselines in every year to 2026, but its error rose from 4.1% in 2019 to 5.7% in 2025 (the last full year). Retraining each year kept the error lower (5.0% in 2025), though it still rose. The 2020 lockdown year seems to have raised the error by about one percentage point, which fell back in 2021, so the longer-term problem was gradual drift, not a single shock. 



##### Data

NESO "Historic Demand Data": half-hourly national demand (ND, in MW), one CSV per year, 2012 to 2026 (258,192 half-hours after feature building), downloaded on 26 October 2026. It is published under the NESO Open Data Licence, and NESO is the source. 2026 covers only part of the year. Raw files are not included in this repository.



##### How to run

pip install -r Requirements.txt

python demand\_forecast.py --data data/



Put the yearly NESO CSV files in data/ first. In Colab, open demand\_forecast\_colab.ipynb, run the cells in order and upload the files when asked. Results have been written to results/.



##### Method

* Forecast: demand for each half-hour, using only information at least 24 hours old; demand at the same time on each of the previous seven days (the previous day, two days before and a week before, and the seven-day average), plus time of day, day of week and time of year.
* Baselines: same time yesterday, same time last week, and the average of the same time over the last seven days.
* Model: HistGradientBoostingRegressor (300 iterations, learning rate 0.05, 15 leaves, L2 = 1.0, seed 0). No tuning was done
* Frozen versus retrained: the frozen model is trained once on 2012-2018. The retrained model is refitted for each test year on all earlier years.
* Metric: WAPE (total absolute error divided by total demand), by test year. 



##### Results

|Test year|Mean demand (MW)|Same time yesterday|Same time last week|Mean of last 7 days|Frozen model|Retrained yearly|
|-|-|-|-|-|-|-|
|2019|29,360|6.9%|6.6%|7.2%|4.1%|4.1%|
|2020|27,160|6.8%|7.2%|7.3%|5.1%|4.9%|
|2021|28,350|6.2%|7.0%|6.6%|4.3%|4.2%|
|2022|26,624|7.0%|7.6%|7.6%|5.0%|4.5%|
|2023|26,054|7.0%|7.9%|7.4%|5.5%|4.8%|
|2024|26,291|7.2%|7.7%|7.7%|5.3%|4.7%|
|2025|26,161|7.1%|8.3%|7.5%|5.7%|5.0%|
|2026 (part year)|25,195|7.8%|8.3%|7.7%|6.5%|5.4%|



The 2019 values for the two models are identical because both were trained on data up to 2018.

* Both gradient boosting models beat all three baselines in every year.
* The frozen model's error rose from 4.1% (2019) to 5.7% (2025). The retrained model's error rose from 4.1% to 5.0%.
* The gap between frozen and retrained was 0.2 percentage points in 2020 and 0.5 in 2022, and reached 0.7 in 2025.
* In 2020, the error rose by about one point for both models, then returned close to 2019 levels in 2021 (4.3% and 4.2%).
* Mean demand fell from about 29.4 GW in 2019 to about 26.2 GW in 2025.
* Percentage error rises partly because demand fell; the frozen model's error went from about 1,200 MW in 2019 to about 1,640 MW in 2026, and the retrained model's to about 1,360 MW.



##### Limitations

* 2026 is a partial year. 

&#x09;It leaves out some of the higher-demand winter months, so its error is not comparable with a 	full year. The headline numbers use 2025.

* No weather data. 

&#x09;Temperature strongly affects demand, so this model is missing an important input. The 	remaining error is probably a mix of weather and other changes I did not model.

* No bank holiday or school holiday information.
* Cause of the drift is untested. 

&#x09;Possible reasons include more rooftop solar and other embedded generation (which ND does not 	include), changes in how people use electricity, and the lockdown years, but I did not test 	any of them.

* The definition of ND may have changed over the years, and settlement periods are treated as fixed half-hours, which is approximate on clock-change days.
* One dataset, one split, fixed model settings, and no tests of statistical significance. 

&#x09;The differences between frozen and retrained are consistent across years, but I have not 	measured their uncertainty.

* Simple baselines. 

&#x09;They do not account for weekends, so they are a low bar.

* Not a production forecasting system.



##### Next steps

* Add temperature and bank holiday data and test whether the error stops rising.
* Compare retraining every month with retraining every year.
* Test the cause of the drift by checking error by season and time of day.
* Add an uncertainty range to the forecasts.



##### Files

|Files|Purpose|
|-|-|
|demand\_forecast.py|Data loading, features, models, evaluation and charts|
|demand\_forecast\_colab.ipynb|Same code as a Colab notebook|
|Requirements.txt|Python dependencies|
|results/wape\_by\_year.csv|Error by approach and test year|
|results/error\_by\_year.png,<br />results/mean\_demand\_by\_year.png|Charts|



Data: NESO Historic Demand Data, NESO Open Data Licence

