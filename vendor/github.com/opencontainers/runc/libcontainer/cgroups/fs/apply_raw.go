// +build linux

package fs

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	libcontainerUtils "github.com/opencontainers/runc/libcontainer/utils"
)

var (
	subsystems = subsystemSet{
		&CpusetGroup{},
		&DevicesGroup{},
		&MemoryGroup{},
		&CpuGroup{},
		&CpuacctGroup{},
		&PidsGroup{},
		&BlkioGroup{},
		&HugetlbGroup{},
		&NetClsGroup{},
		&NetPrioGroup{},
		&PerfEventGroup{},
		&FreezerGroup{},
		&NameGroup{GroupName: "name=systemd", Join: true},
	}
	HugePageSizes, _ = cgroups.GetHugePageSize()
)

var errSubsystemDoesNotExist = errors.New("cgroup: subsystem does not exist")

type subsystemSet []subsystem

func (s subsystemSet) Get(name string) (subsystem, error) {
	for _, ss := range s {
		if ss.Name() == name {
			return ss, nil
		}
	}
	return nil, errSubsystemDoesNotExist
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// Returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
	// Removes the cgroup represented by 'cgroupData'.
	Remove(*cgroupData) error
	// Creates and joins the cgroup represented by 'cgroupData'.
	Apply(*cgroupData) error
	// Set the cgroup represented by cgroup.
	Set(path string, cgroup *configs.Cgroup) error
}

type Manager struct {
	mu      sync.Mutex
	Cgroups *configs.Cgroup
	Paths   map[string]string
}

// The absolute path to the root of the cgroup hierarchies.
var cgroupRootLock sync.Mutex
var cgroupRoot string

// Gets the cgroupRoot.
func getCgroupRoot() (string, error) {
	cgroupRootLock.Lock()
	defer cgroupRootLock.Unlock()

	if cgroupRoot != "" {
		return cgroupRoot, nil
	}

	root, err := cgroups.FindCgroupMountpointDir()
	if err != nil {
		return "", err
	}

	if _, err := os.Stat(root); err != nil {
		return "", err
	}

	cgroupRoot = root
	return cgroupRoot, nil
}

type cgroupData struct {
	root      string
	innerPath string
	config    *configs.Cgroup
	pid       int
}

func (m *Manager) Apply(pid int) (err error) {
	if m.Cgroups == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	var c = m.Cgroups

	d, err := getCgroupData(m.Cgroups, pid)
	if err != nil {
		return err
	}

	m.Paths = make(map[string]string)
	if c.Paths != nil {
		for name, path := range c.Paths {
			_, err := d.path(name)
			if err != nil {
				if cgroups.IsNotFound(err) {
					continue
				}
				return err
			}
			m.Paths[name] = path
		}
		return cgroups.EnterPid(m.Paths, pid)
	}

	for _, sys := range subsystems {
		// TODO: Apply should, ideally, be reentrant or be broken up into a separate
		// create and join phase so that the cgroup hierarchy for a container can be
		// created then join consists of writing the process pids to cgroup.procs
		p, err := d.path(sys.Name())
		if err != nil {
			// The non-presence of the devices subsystem is
			// considered fatal for security reasons.
			if cgroups.IsNotFound(err) && sys.Name() != "devices" {
				continue
			}
			return err
		}
		m.Paths[sys.Name()] = p

		if err := sys.Apply(d); err != nil {
			if os.IsPermission(err) && m.Cgroups.Path == "" {
				// If we didn't set a cgroup path, then let's defer the error here
				// until we know whether we have set limits or not.
				// If we hadn't set limits, then it's ok that we couldn't join this cgroup, because
				// it will have the same limits as its parent.
				delete(m.Paths, sys.Name())
				continue
			}
			return err
		}

	}
	return nil
}

func (m *Manager) Destroy() error {
	if m.Cgroups.Paths != nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := cgroups.RemovePaths(m.Paths); err != nil {
		return err
	}
	m.Paths = make(map[string]string)
	return nil
}

func (m *Manager) GetPaths() map[string]string {
	m.mu.Lock()
	paths := m.Paths
	m.mu.Unlock()
	return paths
}

func (m *Manager) GetStats() (*cgroups.Stats, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	stats := cgroups.NewStats()
	for name, path := range m.Paths {
		sys, err := subsystems.Get(name)
		if err == errSubsystemDoesNotExist || !cgroups.PathExists(path) {
			continue
		}
		if err := sys.GetStats(path, stats); err != nil {
			return nil, err
		}
	}
	return stats, nil
}

func (m *Manager) Set(container *configs.Config) error {
	// If Paths are set, then we are just joining cgroups paths
	// and there is no need to set any values.
	if m.Cgroups.Paths != nil {
		return nil
	}

	paths := m.GetPaths()
	for _, sys := range subsystems {
		path := paths[sys.Name()]
		if err := sys.Set(path, container.Cgroups); err != nil {
			if path == "" {
				// cgroup never applied
				return fmt.Errorf("cannot set limits on the %s cgroup, as the container has not joined it", sys.Name())
			}
			return err
		}
	}

	if m.Paths["cpu"] != "" {
		if err := CheckCpushares(m.Paths["cpu"], container.Cgroups.Resources.CpuShares); err != nil {
			return err
		}
	}
	return nil
}

// Freeze toggles the container's freezer cgroup depending on the state
// provided
func (m *Manager) Freeze(state configs.FreezerState) error {
	paths := m.GetPaths()
	dir := paths["freezer"]
	prevState := m.Cgroups.Resources.Freezer
	m.Cgroups.Resources.Freezer = state
	freezer, err := subsystems.Get("freezer")
	if err != nil {
		return err
	}
	err = freezer.Set(dir, m.Cgroups)
	if err != nil {
		m.Cgroups.Resources.Freezer = prevState
		return err
	}
	return nil
}

func (m *Manager) GetPids() ([]int, error) {
	paths := m.GetPaths()
	return cgroups.GetPids(paths["devices"])
}

func (m *Manager) GetAllPids() ([]int, error) {
	paths := m.GetPaths()
	return cgroups.GetAllPids(paths["devices"])
}

func getCgroupData(c *configs.Cgroup, pid int) (*cgroupData, error) {
	root, err := getCgroupRoot()
	if err != nil {
		return nil, err
	}

	if (c.Name != "" || c.Parent != "") && c.Path != "" {
		return nil, fmt.Errorf("cgroup: either Path or Name and Parent should be used")
	}

	// XXX: Do not remove this code. Path safety is important! -- cyphar
	cgPath := libcontainerUtils.CleanPath(c.Path)
	cgParent := libcontainerUtils.CleanPath(c.Parent)
	cgName := libcontainerUtils.CleanPath(c.Name)

	innerPath := cgPath
	if innerPath == "" {
		innerPath = filepath.Join(cgParent, cgName)
	}

	return &cgroupData{
		root:      root,
		innerPath: innerPath,
		config:    c,
		pid:       pid,
	}, nil
}

func (raw *cgroupData) path(subsystem string) (string, error) {
	mnt, err := cgroups.FindCgroupMountpoint(subsystem)
	// If we didn't mount the subsystem, there is no point we make the path.
	if err != nil {
		return "", err
	}

	// If the cgroup name/path is absolute do not look relative to the cgroup of the init process.
	if filepath.IsAbs(raw.innerPath) {
		// Sometimes subsystems can be mounted together as 'cpu,cpuacct'.
		return filepath.Join(raw.root, filepath.Base(mnt), raw.innerPath), nil
	}

	// Use GetOwnCgroupPath instead of GetInitCgroupPath, because the creating
	// process could in container and shared pid namespace with host, and
	// /proc/1/cgroup could point to whole other world of cgroups.
	parentPath, err := cgroups.GetOwnCgroupPath(subsystem)
	if err != nil {
		return "", err
	}

	return filepath.Join(parentPath, raw.innerPath), nil
}

func (raw *cgroupData) join(subsystem string) (string, error) {
	path, err := raw.path(subsystem)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(path, 0755); err != nil {
		return "", err
	}
	if err := cgroups.WriteCgroupProc(path, raw.pid); err != nil {
		return "", err
	}
	return path, nil
}

func writeFile(dir, file, data string) error {
	// Normally dir should not be empty, one case is that cgroup subsystem
	// is not mounted, we will get empty dir, and we want it fail here.
	if dir == "" {
		return fmt.Errorf("no such directory for %s", file)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, file), []byte(data), 0700); err != nil {
		return fmt.Errorf("failed to write %v to %v: %v", data, file, err)
	}
	return nil
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

func CheckCpushares(path string, c uint64) error {
	var cpuShares uint64

	if c == 0 {
		return nil
	}

	fd, err := os.Open(filepath.Join(path, "cpu.shares"))
	if err != nil {
		return err
	}
	defer fd.Close()

	_, err = fmt.Fscanf(fd, "%d", &cpuShares)
	if err != nil && err != io.EOF {
		return err
	}

	if c > cpuShares {
		return fmt.Errorf("The maximum allowed cpu-shares is %d", cpuShares)
	} else if c < cpuShares {
		return fmt.Errorf("The minimum allowed cpu-shares is %d", cpuShares)
	}

	return nil
}
