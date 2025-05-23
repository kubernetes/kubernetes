package systemd

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	"github.com/sirupsen/logrus"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fs"
)

type LegacyManager struct {
	mu      sync.Mutex
	cgroups *cgroups.Cgroup
	paths   map[string]string
	dbus    *dbusConnManager
}

func NewLegacyManager(cg *cgroups.Cgroup, paths map[string]string) (*LegacyManager, error) {
	if cg.Rootless {
		return nil, errors.New("cannot use rootless systemd cgroups manager on cgroup v1")
	}
	if cg.Resources != nil && cg.Resources.Unified != nil {
		return nil, cgroups.ErrV1NoUnified
	}
	if paths == nil {
		var err error
		paths, err = initPaths(cg)
		if err != nil {
			return nil, err
		}
	}
	return &LegacyManager{
		cgroups: cg,
		paths:   paths,
		dbus:    newDbusConnManager(false),
	}, nil
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// GetStats returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
	// Set sets cgroup resource limits.
	Set(path string, r *cgroups.Resources) error
}

var errSubsystemDoesNotExist = errors.New("cgroup: subsystem does not exist")

var legacySubsystems = []subsystem{
	&fs.CpusetGroup{},
	&fs.DevicesGroup{},
	&fs.MemoryGroup{},
	&fs.CpuGroup{},
	&fs.CpuacctGroup{},
	&fs.PidsGroup{},
	&fs.BlkioGroup{},
	&fs.HugetlbGroup{},
	&fs.PerfEventGroup{},
	&fs.FreezerGroup{},
	&fs.NetPrioGroup{},
	&fs.NetClsGroup{},
	&fs.NameGroup{GroupName: "name=systemd"},
	&fs.RdmaGroup{},
	&fs.NameGroup{GroupName: "misc"},
}

func genV1ResourcesProperties(r *cgroups.Resources, cm *dbusConnManager) ([]systemdDbus.Property, error) {
	var properties []systemdDbus.Property

	deviceProperties, err := generateDeviceProperties(r, cm)
	if err != nil {
		return nil, err
	}
	properties = append(properties, deviceProperties...)

	if r.Memory != 0 {
		properties = append(properties,
			newProp("MemoryLimit", uint64(r.Memory)))
	}

	if r.CpuShares != 0 {
		properties = append(properties,
			newProp("CPUShares", r.CpuShares))
	}

	addCPUQuota(cm, &properties, &r.CpuQuota, r.CpuPeriod)

	if r.BlkioWeight != 0 {
		properties = append(properties,
			newProp("BlockIOWeight", uint64(r.BlkioWeight)))
	}

	if r.PidsLimit > 0 || r.PidsLimit == -1 {
		properties = append(properties,
			newProp("TasksMax", uint64(r.PidsLimit)))
	}

	err = addCpuset(cm, &properties, r.CpusetCpus, r.CpusetMems)
	if err != nil {
		return nil, err
	}

	return properties, nil
}

// initPaths figures out and returns paths to cgroups.
func initPaths(c *cgroups.Cgroup) (map[string]string, error) {
	slice := "system.slice"
	if c.Parent != "" {
		var err error
		slice, err = ExpandSlice(c.Parent)
		if err != nil {
			return nil, err
		}
	}

	unit := getUnitName(c)

	paths := make(map[string]string)
	for _, s := range legacySubsystems {
		subsystemPath, err := getSubsystemPath(slice, unit, s.Name())
		if err != nil {
			// Even if it's `not found` error, we'll return err
			// because devices cgroup is hard requirement for
			// container security.
			if s.Name() == "devices" {
				return nil, err
			}
			// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
			if cgroups.IsNotFound(err) {
				continue
			}
			return nil, err
		}
		paths[s.Name()] = subsystemPath
	}

	// If systemd is using cgroups-hybrid mode then add the slice path of
	// this container to the paths so the following process executed with
	// "runc exec" joins that cgroup as well.
	if cgroups.IsCgroup2HybridMode() {
		// "" means cgroup-hybrid path
		cgroupsHybridPath, err := getSubsystemPath(slice, unit, "")
		if err != nil && cgroups.IsNotFound(err) {
			return nil, err
		}
		paths[""] = cgroupsHybridPath
	}

	return paths, nil
}

func (m *LegacyManager) Apply(pid int) error {
	var (
		c          = m.cgroups
		unitName   = getUnitName(c)
		slice      = "system.slice"
		properties []systemdDbus.Property
	)

	m.mu.Lock()
	defer m.mu.Unlock()

	if c.Parent != "" {
		slice = c.Parent
	}

	properties = append(properties, systemdDbus.PropDescription("libcontainer container "+c.Name))

	if strings.HasSuffix(unitName, ".slice") {
		// If we create a slice, the parent is defined via a Wants=.
		properties = append(properties, systemdDbus.PropWants(slice))
	} else {
		// Otherwise it's a scope, which we put into a Slice=.
		properties = append(properties, systemdDbus.PropSlice(slice))
		// Assume scopes always support delegation (supported since systemd v218).
		properties = append(properties, newProp("Delegate", true))
	}

	// only add pid if its valid, -1 is used w/ general slice creation.
	if pid != -1 {
		properties = append(properties, newProp("PIDs", []uint32{uint32(pid)}))
	}

	// Always enable accounting, this gets us the same behaviour as the fs implementation,
	// plus the kernel has some problems with joining the memory cgroup at a later time.
	properties = append(properties,
		newProp("MemoryAccounting", true),
		newProp("CPUAccounting", true),
		newProp("BlockIOAccounting", true),
		newProp("TasksAccounting", true),
	)

	// Assume DefaultDependencies= will always work (the check for it was previously broken.)
	properties = append(properties,
		newProp("DefaultDependencies", false))

	properties = append(properties, c.SystemdProps...)

	if err := startUnit(m.dbus, unitName, properties, pid == -1); err != nil {
		return err
	}

	if err := m.joinCgroups(pid); err != nil {
		return err
	}

	return nil
}

func (m *LegacyManager) Destroy() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	stopErr := stopUnit(m.dbus, getUnitName(m.cgroups))

	// Both on success and on error, cleanup all the cgroups
	// we are aware of, as some of them were created directly
	// by Apply() and are not managed by systemd.
	if err := cgroups.RemovePaths(m.paths); err != nil && stopErr == nil {
		return err
	}

	return stopErr
}

func (m *LegacyManager) Path(subsys string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths[subsys]
}

func (m *LegacyManager) joinCgroups(pid int) error {
	for _, sys := range legacySubsystems {
		name := sys.Name()
		switch name {
		case "name=systemd":
			// let systemd handle this
		case "cpuset":
			if path, ok := m.paths[name]; ok {
				s := &fs.CpusetGroup{}
				if err := s.ApplyDir(path, m.cgroups.Resources, pid); err != nil {
					return err
				}
			}
		default:
			if path, ok := m.paths[name]; ok {
				if err := os.MkdirAll(path, 0o755); err != nil {
					return err
				}
				if err := cgroups.WriteCgroupProc(path, pid); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func getSubsystemPath(slice, unit, subsystem string) (string, error) {
	mountpoint, err := cgroups.FindCgroupMountpoint("", subsystem)
	if err != nil {
		return "", err
	}

	return filepath.Join(mountpoint, slice, unit), nil
}

func (m *LegacyManager) Freeze(state cgroups.FreezerState) error {
	err := m.doFreeze(state)
	if err == nil {
		m.cgroups.Resources.Freezer = state
	}
	return err
}

// doFreeze is the same as Freeze but without
// changing the m.cgroups.Resources.Frozen field.
func (m *LegacyManager) doFreeze(state cgroups.FreezerState) error {
	path, ok := m.paths["freezer"]
	if !ok {
		return errSubsystemDoesNotExist
	}
	freezer := &fs.FreezerGroup{}
	resources := &cgroups.Resources{Freezer: state}
	return freezer.Set(path, resources)
}

func (m *LegacyManager) GetPids() ([]int, error) {
	path, ok := m.paths["devices"]
	if !ok {
		return nil, errSubsystemDoesNotExist
	}
	return cgroups.GetPids(path)
}

func (m *LegacyManager) GetAllPids() ([]int, error) {
	path, ok := m.paths["devices"]
	if !ok {
		return nil, errSubsystemDoesNotExist
	}
	return cgroups.GetAllPids(path)
}

func (m *LegacyManager) GetStats() (*cgroups.Stats, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	stats := cgroups.NewStats()
	for _, sys := range legacySubsystems {
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

func (m *LegacyManager) Set(r *cgroups.Resources) error {
	if r == nil {
		return nil
	}
	if r.Unified != nil {
		return cgroups.ErrV1NoUnified
	}
	// Use a copy since CpuQuota in r may be modified.
	rCopy := *r
	r = &rCopy
	properties, err := genV1ResourcesProperties(r, m.dbus)
	if err != nil {
		return err
	}

	unitName := getUnitName(m.cgroups)
	needsFreeze, needsThaw, err := m.freezeBeforeSet(unitName, r)
	if err != nil {
		return err
	}

	if needsFreeze {
		if err := m.doFreeze(cgroups.Frozen); err != nil {
			// If freezer cgroup isn't supported, we just warn about it.
			logrus.Infof("freeze container before SetUnitProperties failed: %v", err)
			// skip update the cgroup while frozen failed. #3803
			if !errors.Is(err, errSubsystemDoesNotExist) {
				if needsThaw {
					if thawErr := m.doFreeze(cgroups.Thawed); thawErr != nil {
						logrus.Infof("thaw container after doFreeze failed: %v", thawErr)
					}
				}
				return err
			}
		}
	}
	setErr := setUnitProperties(m.dbus, unitName, properties...)
	if needsThaw {
		if err := m.doFreeze(cgroups.Thawed); err != nil {
			logrus.Infof("thaw container after SetUnitProperties failed: %v", err)
		}
	}
	if setErr != nil {
		return setErr
	}

	for _, sys := range legacySubsystems {
		// Get the subsystem path, but don't error out for not found cgroups.
		path, ok := m.paths[sys.Name()]
		if !ok {
			continue
		}
		if err := sys.Set(path, r); err != nil {
			return err
		}
	}

	return nil
}

func (m *LegacyManager) GetPaths() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths
}

func (m *LegacyManager) GetCgroups() (*cgroups.Cgroup, error) {
	return m.cgroups, nil
}

func (m *LegacyManager) GetFreezerState() (cgroups.FreezerState, error) {
	path, ok := m.paths["freezer"]
	if !ok {
		return cgroups.Undefined, nil
	}
	freezer := &fs.FreezerGroup{}
	return freezer.GetState(path)
}

func (m *LegacyManager) Exists() bool {
	return cgroups.PathExists(m.Path("devices"))
}

func (m *LegacyManager) OOMKillCount() (uint64, error) {
	return fs.OOMKillCount(m.Path("memory"))
}
