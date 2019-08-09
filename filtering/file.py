import numpy as np
import h5py
import tempfile


class LagrangeParticleFile(object):
    """A specialised ParticleFile class for efficient input/output of
    temporary particle data.

    All variables that are marked "to_write" are written to a temporary HDF5 file.
    """

    def __init__(self, particleset, outputdt=np.infty, variables=None):
        # required by parcels
        self.outputdt = outputdt

        self.n = len(particleset)

        self._tempfile = tempfile.NamedTemporaryFile(dir=".", suffix=".h5")
        self.h5file = h5py.File(self._tempfile)

        # variable -> dtype map for creating datasets
        self._vars = {}

        for v in particleset.ptype.variables:
            # this variable isn't marked for output to file -- honour that
            if not v.to_write:
                continue

            # there's an explicit list of variables for us to write, so
            # filter based on that (e.g. outputting only sample_variables)
            if variables is not None and v.name not in variables:
                continue

            self._vars[v.name] = v.dtype

        # create empty time attribute
        self.h5file.attrs["time"] = []

    def set_group(self, group):
        """Set the group for subsequent write operations.

        Creates the group, and datasets for all written variables
        if they do not already exist.
        """

        self._group = self.h5file.require_group(group)
        self._var_datasets = {}
        for v, t in self._vars.items():
            self._var_datasets[v] = self._group.require_dataset(
                v, shape=(0, self.n), maxshape=(None, self.n), dtype=t
            )

    def data(self, group):
        """Return the underlying group from the HDF5 object."""

        return self.h5file.require_group(group)

    def write(self, particleset, time, deleted_only=False):
        """Write particle data in the particleset at time to this ParticleFile's temporary dataset."""

        # don't write out deleted particles
        if deleted_only:
            return

        self.h5file.attrs["time"] = np.append(self.h5file.attrs["time"], time)

        for v, d in self._var_datasets.items():
            # first, resize all datasets to add another entry in the time dimension
            # then we can just pull the array for this variable out of the particleset
            d.resize(d.shape[0] + 1, axis=0)
            d[-1, :] = particleset.particle_data[v]


class LagrangeParticleBuffer(object):
    """A specialised ParticleFile class for in-memory buffering of temporary particle data."""

    def __init__(self, particleset, outputdt=np.infty, variables=None):
        self.outputdt = outputdt
        self.n = len(particleset)

        # get variable names
        self._vars = {
            v.name: v.dtype for v in particleset.ptype.variables if v.to_write
        }
        if variables is not None:
            # filter based on passed-in list, if required
            self._vars = {k: v for k, v in self._vars.items() if k in variables}

        self._groups = {}

    def set_group(self, group):
        """Set the group for subsequent write operations."""

        # has this group been requested already?
        if group in self._groups:
            self._var_datasets = self._groups[group]
            return

        # otherwise, create new empty arrays to store all variables
        self._var_datasets = {}
        for v, t in self._vars.items():
            self._var_datasets[v] = np.empty((0, self.n), t)

        self._groups[group] = self._var_datasets

    def data(self, group):
        """Get the underlying data for the given group."""

        return self._groups[group]

    def write(self, particleset, time, deleted_only=False):
        """Write particle data in the particleset at time to this ParticleFile's temporary dataset."""

        if deleted_only:
            return

        for v in self._var_datasets:
            self._var_datasets[v] = np.concatenate(
                (self._var_datasets[v], particleset.particle_data[v][None, :]), axis=0
            )
