# Predicting Car Crash Severity for Emergency Vehicle Response


## Working repo for my Data Science Immersive capstone at Galvanize  |  Aug - Sept. 2017

## The data
From New Jersey state crash data.
http://www.state.nj.us/transportation/refdata/accident/


#### Capstone presentation 9/7/17
https://docs.google.com/presentation/d/1L-wzFSAEA7NZ9pj0hQhlBhd22jeJlCCqQHXwVjCY7Ig/edit?usp=sharing



### The problem/question:
- Classify car crashes as involving injury or not
- In its real world use case, an 911 dispatcher would use this model to inform her decision to send or not send emergency medical care.


Business understanding. On first guess, what would the features likely be?
*Given that you already know there's been a crash*

### Physical Impact Conditions:
- Direction of the other vehicles
- Where the impact was on body of car
- Speed
- Number of vehicles involved
- Size of vehicles relative to one another

### The Environment Conditions:
- Deadly cliffs, mountains (bridges?)
- Time of day
- Day of week
- Type of road
- Weather
- Atmospheric cover on the road

### Conditions About the Occupants:
- Weight and age of people involved
- Use of seat belts
- Pedestrians or cyclists involved?
- Trucks involved?
- Was there an illegal maneuver involved?

### Cars:
- Vehicle safety rating


## Objectives and process:
- Engineer that data, aggregating by case number to gain additional insights than the raw data provides.
- Drop the data that doesn't carry signal or improve the model.
- Improve model accuracy by trying new features and continue to iterate to improve it.
- Tweak parameters
- Grid search for a more exhaustive exploration of parameters
- Validate and consider other alternatives.
