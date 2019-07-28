# lagrangian-filtering
Temporal filtering of data in a Lagrangian frame of reference.

## Overview
This provides a library and a post-processing analysis tool for the
removal of sub-inertial frequencies from data in a Lagrangian frame of
reference. This may be useful, for example, for distinguishing
stationary internal waves. At a high level, the algorithm looks like:

1. particle advection using [OceanParcels](http://oceanparcels.org)
2. sampling of data (e.g. velocity, density) along particle tracks
3. temporal filtering
4. writing filtered data back to disk

## Installation
For the moment, the easiest way to get started is using Python's `pip`
package manager. Optionally, the package can be installed inside a
`virtualenv` virtual environment, for cleaner separation from your
native Python environment. [Conda](https://conda.io) support is
planned, but not yet present. There are very few dependencies, but a
custom branch of OceanParcels is necessary to get acceptable
performance at the moment. During the development phase, I recommend
installing this as a "development package", meaning that changes to
your local checkout are instantly reflected in your Python
environment.

1. Clone this repository `git clone https://github.com/angus-g/lagrangian-filtering`
2. Change to the directory `cd lagrangian-filtering`
3. (Optional) Create the virtualenv `virtualenv env` and activate it `source env/bin/activate`
4. Install the prerequisites `pip install -r requirements.txt`
5. Install the development version of the package `pip install -e .`

### Upgrading
With the package installed in development mode, updating is as easy as
`git pull` (or making local changes) in the `lagrangian-filtering`
directory.

## Usage
For the moment, it's easiest to set up the filtering workflow in a script or
a jupyter notebook. An example looks like:

```python
import filtering
from datetime import timedelta

filenames = {
	"U": "/data/data_wave_U.nc", "V": "/data/data_wave_V.nc"
}
variables = {"U": "U", "V": "V"}
dimensions = {"lon": "X", "lat", "Y", "time": "T"}

f = filtering.LagrangeFilter(
	"waves", filenames, variables, dimensions,
	sample_variables=["U"], mesh="flat",
	window_size=timedelta(days=2).total_seconds()
)

f()
```

This uses velocity data from the two specified files. Zonal velocity
data from the `U` variable will be sampled and filtered, with a
filtering window for 2 days on either side of each sample (i.e. a
4-day window for filtering).
