package fs

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

var subsystems = []subsystem{
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
	&RdmaGroup{},
	&NameGroup{GroupName: "name=systemd", Join: true},
}

var errSubsystemDoesNotExist = errors.New("cgroup: subsystem does not exist")

func init() {
	// If using cgroups-hybrid mode then add a "" controller indicating
	// it should join the cgroups v2.
	if cgroups.IsCgroup2HybridMode() {
		subsystems = append(subsystems, &NameGroup{GroupName: "", Join: true})
	}
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// GetStats fills in the stats for the subsystem.
	GetStats(path string, stats *cgroups.Stats) error
	// Apply creates and joins a cgroup, adding pid into it. Some
	// subsystems use resources to pre-configure the cgroup parents
	// before creating or joining it.
	Apply(path string, r *configs.Resources, pid int) error
	// Set sets the cgroup resources.
	Set(path string, r *configs.Resources) error
}

type manager struct {
	mu      sync.Mutex
	cgroups *configs.Cgroup
	paths   map[string]string
}

func NewManager(cg *configs.Cgroup, paths map[string]string) (cgroups.Manager, error) {
	// Some v1 controllers (cpu, cpuset, and devices) expect
	// cgroups.Resources to not be nil in Apply.
	if cg.Resources == nil {
		return nil, errors.New("cgroup v1 manager needs configs.Resources to be set during manager creation")
	}
	if cg.Resources.Unified != nil {
		return nil, cgroups.ErrV1NoUnified
	}

	if paths == nil {
		var err error
		paths, err = initPaths(cg)
		if err != nil {
			return nil, err
		}
	}

	return &manager{
		cgroups: cg,
		paths:   paths,
	}, nil
}

// isIgnorableError returns whether err is a permission error (in the loose
// sense of the word). This includes EROFS (which for an unprivileged user is
// basically a permission error) and EACCES (for similar reasons) as well as
// the normal EPERM.
func isIgnorableError(rootless bool, err error) bool {
	// We do not ignore errors if we are root.
	if !rootless {
		return false
	}
	// Is it an ordinary EPERM?
	if errors.Is(err, os.ErrPermission) {
		return true
	}
	// Handle some specific syscall errors.
	var errno unix.Errno
	if errors.As(err, &errno) {
		return errno == unix.EROFS || errno == unix.EPERM || errno == unix.EACCES
	}
	return false
}

func (m *manager) Apply(pid int) (err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	c := m.cgroups

	for _, sys := range subsystems {
		name := sys.Name()
		p, ok := m.paths[name]
		if !ok {
			continue
		}

		if err := sys.Apply(p, c.Resources, pid); err != nil {
			// In the case of rootless (including euid=0 in userns), where an
			// explicit cgroup path hasn't been set, we don't bail on error in
			// case of permission problems here, but do delete the path from
			// the m.paths map, since it is either non-existent and could not
			// be created, or the pid could not be added to it.
			//
			// Cases where limits for the subsystem have been set are handled
			// later by Set, which fails with a friendly error (see
			// if path == "" in Set).
			if isIgnorableError(c.Rootless, err) && c.Path == "" {
				delete(m.paths, name)
				continue
			}
			return err
		}

	}
	return nil
}

func (m *manager) Destroy() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return cgroups.RemovePaths(m.paths)
}

func (m *manager) Path(subsys string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths[subsys]
}

func (m *manager) GetStats() (*cgroups.Stats, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	stats := cgroups.NewStats()
	for _, sys := range subsystems {
		path := m.paths[sys.Name()]
		if path == "" {
			continue
		}
		if err := sys.GetStats(path, stats); err != nil {
			return nil, err
		}
	}
	return stats, nil
}

func (m *manager) Set(r *configs.Resources) error {
	if r == nil {
		return nil
	}

	if r.Unified != nil {
		return cgroups.ErrV1NoUnified
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, sys := range subsystems {
		path := m.paths[sys.Name()]
		if err := sys.Set(path, r); err != nil {
			// When rootless is true, errors from the device subsystem
			// are ignored, as it is really not expected to work.
			if m.cgroups.Rootless && sys.Name() == "devices" {
				continue
			}
			// However, errors from other subsystems are not ignored.
			// see @test "runc create (rootless + limits + no cgrouppath + no permission) fails with informative error"
			if path == "" {
				// We never created a path for this cgroup, so we cannot set
				// limits for it (though we have already tried at this point).
				return fmt.Errorf("cannot set %s limit: container could not join or create cgroup", sys.Name())
			}
			return err
		}
	}

	return nil
}

// Freeze toggles the container's freezer cgroup depending on the state
// provided
func (m *manager) Freeze(state configs.FreezerState) error {
	path := m.Path("freezer")
	if path == "" {
		return errors.New("cannot toggle freezer: cgroups not configured for container")
	}

	prevState := m.cgroups.Resources.Freezer
	m.cgroups.Resources.Freezer = state
	freezer := &FreezerGroup{}
	if err := freezer.Set(path, m.cgroups.Resources); err != nil {
		m.cgroups.Resources.Freezer = prevState
		return err
	}
	return nil
}

func (m *manager) GetPids() ([]int, error) {
	return cgroups.GetPids(m.Path("devices"))
}

func (m *manager) GetAllPids() ([]int, error) {
	return cgroups.GetAllPids(m.Path("devices"))
}

func (m *manager) GetPaths() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths
}

func (m *manager) GetCgroups() (*configs.Cgroup, error) {
	return m.cgroups, nil
}

func (m *manager) GetFreezerState() (configs.FreezerState, error) {
	dir := m.Path("freezer")
	// If the container doesn't have the freezer cgroup, say it's undefined.
	if dir == "" {
		return configs.Undefined, nil
	}
	freezer := &FreezerGroup{}
	return freezer.GetState(dir)
}

func (m *manager) Exists() bool {
	return cgroups.PathExists(m.Path("devices"))
}

func OOMKillCount(path string) (uint64, error) {
	return fscommon.GetValueByKey(path, "memory.oom_control", "oom_kill")
}

func (m *manager) OOMKillCount() (uint64, error) {
	c, err := OOMKillCount(m.Path("memory"))
	// Ignore ENOENT when rootless as it couldn't create cgroup.
	if err != nil && m.cgroups.Rootless && os.IsNotExist(err) {
		err = nil
	}

	return c, err
}
