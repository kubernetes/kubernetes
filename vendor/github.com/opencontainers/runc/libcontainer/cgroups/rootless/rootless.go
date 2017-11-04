// +build linux

package rootless

import (
	"fmt"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/configs/validate"
)

// TODO: This is copied from libcontainer/cgroups/fs, which duplicates this code
//       needlessly. We should probably export this list.

var subsystems = []subsystem{
	&fs.CpusetGroup{},
	&fs.DevicesGroup{},
	&fs.MemoryGroup{},
	&fs.CpuGroup{},
	&fs.CpuacctGroup{},
	&fs.PidsGroup{},
	&fs.BlkioGroup{},
	&fs.HugetlbGroup{},
	&fs.NetClsGroup{},
	&fs.NetPrioGroup{},
	&fs.PerfEventGroup{},
	&fs.FreezerGroup{},
	&fs.NameGroup{GroupName: "name=systemd"},
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string

	// Returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
}

// The noop cgroup manager is used for rootless containers, because we currently
// cannot manage cgroups if we are in a rootless setup. This manager is chosen
// by factory if we are in rootless mode. We error out if any cgroup options are
// set in the config -- this may change in the future with upcoming kernel features
// like the cgroup namespace.

type Manager struct {
	Cgroups *configs.Cgroup
	Paths   map[string]string
}

func (m *Manager) Apply(pid int) error {
	// If there are no cgroup settings, there's nothing to do.
	if m.Cgroups == nil {
		return nil
	}

	// We can't set paths.
	// TODO(cyphar): Implement the case where the runner of a rootless container
	//               owns their own cgroup, which would allow us to set up a
	//               cgroup for each path.
	if m.Cgroups.Paths != nil {
		return fmt.Errorf("cannot change cgroup path in rootless container")
	}

	// We load the paths into the manager.
	paths := make(map[string]string)
	for _, sys := range subsystems {
		name := sys.Name()

		path, err := cgroups.GetOwnCgroupPath(name)
		if err != nil {
			// Ignore paths we couldn't resolve.
			continue
		}

		paths[name] = path
	}

	m.Paths = paths
	return nil
}

func (m *Manager) GetPaths() map[string]string {
	return m.Paths
}

func (m *Manager) Set(container *configs.Config) error {
	// We have to re-do the validation here, since someone might decide to
	// update a rootless container.
	return validate.New().Validate(container)
}

func (m *Manager) GetPids() ([]int, error) {
	dir, err := cgroups.GetOwnCgroupPath("devices")
	if err != nil {
		return nil, err
	}
	return cgroups.GetPids(dir)
}

func (m *Manager) GetAllPids() ([]int, error) {
	dir, err := cgroups.GetOwnCgroupPath("devices")
	if err != nil {
		return nil, err
	}
	return cgroups.GetAllPids(dir)
}

func (m *Manager) GetStats() (*cgroups.Stats, error) {
	// TODO(cyphar): We can make this work if we figure out a way to allow usage
	//               of cgroups with a rootless container. While this doesn't
	//               actually require write access to a cgroup directory, the
	//               statistics are not useful if they can be affected by
	//               non-container processes.
	return nil, fmt.Errorf("cannot get cgroup stats in rootless container")
}

func (m *Manager) Freeze(state configs.FreezerState) error {
	// TODO(cyphar): We can make this work if we figure out a way to allow usage
	//               of cgroups with a rootless container.
	return fmt.Errorf("cannot use freezer cgroup in rootless container")
}

func (m *Manager) Destroy() error {
	// We don't have to do anything here because we didn't do any setup.
	return nil
}
