# Route Optimizer for Property Engineers

This Python app optimizes routes for property engineers visiting multiple locations throughout their workday. It considers real-world constraints like service durations, time windows, and priorities while still minimizing total travel distance.

## Installation

After cloning the repo,
1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The system accepts location data in a simple CSV format:
```
x,y
0,0
10,20
15,30
```

Optionally, you can add a `time_window_start` and `time_window_end` to each location, like:
```
x,y,time_window_start,time_window_end
0,0,8:00,10:00
10,20,10:00,12:00
15,30,12:00,14:00
```

By default, the optimizer will consider the start time to be 8:00 AM and the time to complete the route as 8 hours (i.e. a full workday).

To run the app, run the following command:
```bash
python solution.py
```

This will read the locations from `delivery_points.txt` and output the optimized route, arrival times, and total distance.

You can also run the tests by executing:
```bash
python -m unittest test.py
```