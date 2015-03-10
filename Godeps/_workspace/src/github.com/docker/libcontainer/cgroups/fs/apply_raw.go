package fs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"

	"github.com/docker/libcontainer/cgroups"
)

var (
	subsystems = map[string]subsystem{
		"devices":    &DevicesGroup{},
		"memory":     &MemoryGroup{},
		"cpu":        &CpuGroup{},
		"cpuset":     &CpusetGroup{},
		"cpuacct":    &CpuacctGroup{},
		"blkio":      &BlkioGroup{},
		"perf_event": &PerfEventGroup{},
		"freezer":    &FreezerGroup{},
	}
	CgroupProcesses = "cgroup.procs"
)

// The absolute path to the root of the cgroup hierarchies.
var cgroupRoot string

// TODO(vmarmol): Report error here, we'll probably need to wait for the new API.
func init() {
	// we can pick any subsystem to find the root
	cpuRoot, err := cgroups.FindCgroupMountpoint("cpu")
	if err != nil {
		return
	}
	cgroupRoot = filepath.Dir(cpuRoot)

	if _, err := os.Stat(cgroupRoot); err != nil {
		return
	}
}

type subsystem interface {
	// Returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
	// Removes the cgroup represented by 'data'.
	Remove(*data) error
	// Creates and joins the cgroup represented by data.
	Set(*data) error
}

type data struct {
	root   string
	cgroup string
	c      *cgroups.Cgroup
	pid    int
}

func Apply(c *cgroups.Cgroup, pid int) (map[string]string, error) {
	d, err := getCgroupData(c, pid)
	if err != nil {
		return nil, err
	}

	paths := make(map[string]string)
	defer func() {
		if err != nil {
			cgroups.RemovePaths(paths)
		}
	}()
	for name, sys := range subsystems {
		if err := sys.Set(d); err != nil {
			return nil, err
		}
		// FIXME: Apply should, ideally, be reentrant or be broken up into a separate
		// create and join phase so that the cgroup hierarchy for a container can be
		// created then join consists of writing the process pids to cgroup.procs
		p, err := d.path(name)
		if err != nil {
			if cgroups.IsNotFound(err) {
				continue
			}
			return nil, err
		}
		paths[name] = p
	}
	return paths, nil
}

// Symmetrical public function to update device based cgroups.  Also available
// in the systemd implementation.
func ApplyDevices(c *cgroups.Cgroup, pid int) error {
	d, err := getCgroupData(c, pid)
	if err != nil {
		return err
	}

	devices := subsystems["devices"]

	return devices.Set(d)
}

func GetStats(systemPaths map[string]string) (*cgroups.Stats, error) {
	stats := cgroups.NewStats()
	for name, path := range systemPaths {
		sys, ok := subsystems[name]
		if !ok || !cgroups.PathExists(path) {
			continue
		}
		if err := sys.GetStats(path, stats); err != nil {
			return nil, err
		}
	}

	return stats, nil
}

// Freeze toggles the container's freezer cgroup depending on the state
// provided
func Freeze(c *cgroups.Cgroup, state cgroups.FreezerState) error {
	d, err := getCgroupData(c, 0)
	if err != nil {
		return err
	}

	c.Freezer = state

	freezer := subsystems["freezer"]

	return freezer.Set(d)
}

func GetPids(c *cgroups.Cgroup) ([]int, error) {
	d, err := getCgroupData(c, 0)
	if err != nil {
		return nil, err
	}

	dir, err := d.path("devices")
	if err != nil {
		return nil, err
	}

	return cgroups.ReadProcsFile(dir)
}

func getCgroupData(c *cgroups.Cgroup, pid int) (*data, error) {
	if cgroupRoot == "" {
		return nil, fmt.Errorf("failed to find the cgroup root")
	}

	cgroup := c.Name
	if c.Parent != "" {
		cgroup = filepath.Join(c.Parent, cgroup)
	}

	return &data{
		root:   cgroupRoot,
		cgroup: cgroup,
		c:      c,
		pid:    pid,
	}, nil
}

func (raw *data) parent(subsystem string) (string, error) {
	initPath, err := cgroups.GetInitCgroupDir(subsystem)
	if err != nil {
		return "", err
	}
	return filepath.Join(raw.root, subsystem, initPath), nil
}

func (raw *data) path(subsystem string) (string, error) {
	// If the cgroup name/path is absolute do not look relative to the cgroup of the init process.
	if filepath.IsAbs(raw.cgroup) {
		path := filepath.Join(raw.root, subsystem, raw.cgroup)

		if _, err := os.Stat(path); err != nil {
			if os.IsNotExist(err) {
				return "", cgroups.NewNotFoundError(subsystem)
			}

			return "", err
		}

		return path, nil
	}

	parent, err := raw.parent(subsystem)
	if err != nil {
		return "", err
	}

	return filepath.Join(parent, raw.cgroup), nil
}

func (raw *data) join(subsystem string) (string, error) {
	path, err := raw.path(subsystem)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(path, 0755); err != nil && !os.IsExist(err) {
		return "", err
	}
	if err := writeFile(path, CgroupProcesses, strconv.Itoa(raw.pid)); err != nil {
		return "", err
	}
	return path, nil
}

func writeFile(dir, file, data string) error {
	return ioutil.WriteFile(filepath.Join(dir, file), []byte(data), 0700)
}

func readFile(dir, file string) (string, error) {
	data, err := ioutil.ReadFile(filepath.Join(dir, file))
	return string(data), err
}

func removePath(p string, err error) error {
	if err != nil {
		return err
	}
	if p != "" {
		return os.RemoveAll(p)
	}
	return nil
}
