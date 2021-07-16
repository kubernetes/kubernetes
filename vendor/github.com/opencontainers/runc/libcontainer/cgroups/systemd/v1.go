// +build linux

package systemd

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	"github.com/godbus/dbus/v5"
	"github.com/sirupsen/logrus"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type legacyManager struct {
	mu      sync.Mutex
	cgroups *configs.Cgroup
	paths   map[string]string
	dbus    *dbusConnManager
}

func NewLegacyManager(cg *configs.Cgroup, paths map[string]string) cgroups.Manager {
	return &legacyManager{
		cgroups: cg,
		paths:   paths,
		dbus:    newDbusConnManager(false),
	}
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// Returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
	// Set sets cgroup resource limits.
	Set(path string, r *configs.Resources) error
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
}

func genV1ResourcesProperties(r *configs.Resources, cm *dbusConnManager) ([]systemdDbus.Property, error) {
	var properties []systemdDbus.Property

	deviceProperties, err := generateDeviceProperties(r)
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

	addCpuQuota(cm, &properties, r.CpuQuota, r.CpuPeriod)

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

func (m *legacyManager) Apply(pid int) error {
	var (
		c          = m.cgroups
		unitName   = getUnitName(c)
		slice      = "system.slice"
		properties []systemdDbus.Property
	)

	if c.Resources.Unified != nil {
		return cgroups.ErrV1NoUnified
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if c.Paths != nil {
		paths := make(map[string]string)
		cgMap, err := cgroups.ParseCgroupFile("/proc/self/cgroup")
		if err != nil {
			return err
		}
		// XXX(kolyshkin@): why this check is needed?
		for name, path := range c.Paths {
			if _, ok := cgMap[name]; ok {
				paths[name] = path
			}
		}
		m.paths = paths
		return cgroups.EnterPid(m.paths, pid)
	}

	if c.Parent != "" {
		slice = c.Parent
	}

	properties = append(properties, systemdDbus.PropDescription("libcontainer container "+c.Name))

	// if we create a slice, the parent is defined via a Wants=
	if strings.HasSuffix(unitName, ".slice") {
		properties = append(properties, systemdDbus.PropWants(slice))
	} else {
		// otherwise, we use Slice=
		properties = append(properties, systemdDbus.PropSlice(slice))
	}

	// only add pid if its valid, -1 is used w/ general slice creation.
	if pid != -1 {
		properties = append(properties, newProp("PIDs", []uint32{uint32(pid)}))
	}

	// Check if we can delegate. This is only supported on systemd versions 218 and above.
	if !strings.HasSuffix(unitName, ".slice") {
		// Assume scopes always support delegation.
		properties = append(properties, newProp("Delegate", true))
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

	if err := startUnit(m.dbus, unitName, properties); err != nil {
		return err
	}

	paths := make(map[string]string)
	for _, s := range legacySubsystems {
		subsystemPath, err := getSubsystemPath(m.cgroups, s.Name())
		if err != nil {
			// Even if it's `not found` error, we'll return err
			// because devices cgroup is hard requirement for
			// container security.
			if s.Name() == "devices" {
				return err
			}
			// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
			if cgroups.IsNotFound(err) {
				continue
			}
			return err
		}
		paths[s.Name()] = subsystemPath
	}
	m.paths = paths

	if err := m.joinCgroups(pid); err != nil {
		return err
	}

	return nil
}

func (m *legacyManager) Destroy() error {
	if m.cgroups.Paths != nil {
		return nil
	}
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

func (m *legacyManager) Path(subsys string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths[subsys]
}

func (m *legacyManager) joinCgroups(pid int) error {
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

func getSubsystemPath(c *configs.Cgroup, subsystem string) (string, error) {
	mountpoint, err := cgroups.FindCgroupMountpoint("", subsystem)
	if err != nil {
		return "", err
	}

	initPath, err := cgroups.GetInitCgroup(subsystem)
	if err != nil {
		return "", err
	}
	// if pid 1 is systemd 226 or later, it will be in init.scope, not the root
	initPath = strings.TrimSuffix(filepath.Clean(initPath), "init.scope")

	slice := "system.slice"
	if c.Parent != "" {
		slice = c.Parent
	}

	slice, err = ExpandSlice(slice)
	if err != nil {
		return "", err
	}

	return filepath.Join(mountpoint, initPath, slice, getUnitName(c)), nil
}

func (m *legacyManager) Freeze(state configs.FreezerState) error {
	err := m.doFreeze(state)
	if err == nil {
		m.cgroups.Resources.Freezer = state
	}
	return err
}

// doFreeze is the same as Freeze but without
// changing the m.cgroups.Resources.Frozen field.
func (m *legacyManager) doFreeze(state configs.FreezerState) error {
	path, ok := m.paths["freezer"]
	if !ok {
		return errSubsystemDoesNotExist
	}
	freezer := &fs.FreezerGroup{}
	resources := &configs.Resources{Freezer: state}
	return freezer.Set(path, resources)
}

func (m *legacyManager) GetPids() ([]int, error) {
	path, ok := m.paths["devices"]
	if !ok {
		return nil, errSubsystemDoesNotExist
	}
	return cgroups.GetPids(path)
}

func (m *legacyManager) GetAllPids() ([]int, error) {
	path, ok := m.paths["devices"]
	if !ok {
		return nil, errSubsystemDoesNotExist
	}
	return cgroups.GetAllPids(path)
}

func (m *legacyManager) GetStats() (*cgroups.Stats, error) {
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

// freezeBeforeSet answers whether there is a need to freeze the cgroup before
// applying its systemd unit properties, and thaw after, while avoiding
// unnecessary freezer state changes.
//
// The reason why we have to freeze is that systemd's application of device
// rules is done disruptively, resulting in spurious errors to common devices
// (unlike our fs driver, they will happily write deny-all rules to running
// containers). So we have to freeze the container to avoid the container get
// an occasional "permission denied" error.
func (m *legacyManager) freezeBeforeSet(unitName string, r *configs.Resources) (needsFreeze, needsThaw bool, err error) {
	// Special case for SkipDevices, as used by Kubernetes to create pod
	// cgroups with allow-all device policy).
	if r.SkipDevices {
		if r.SkipFreezeOnSet {
			// Both needsFreeze and needsThaw are false.
			return
		}

		// No need to freeze if SkipDevices is set, and either
		// (1) systemd unit does not (yet) exist, or
		// (2) it has DevicePolicy=auto and empty DeviceAllow list.
		//
		// Interestingly, (1) and (2) are the same here because
		// a non-existent unit returns default properties,
		// and settings in (2) are the defaults.
		//
		// Do not return errors from getUnitTypeProperty, as they alone
		// should not prevent Set from working.

		unitType := getUnitType(unitName)

		devPolicy, e := getUnitTypeProperty(m.dbus, unitName, unitType, "DevicePolicy")
		if e == nil && devPolicy.Value == dbus.MakeVariant("auto") {
			devAllow, e := getUnitTypeProperty(m.dbus, unitName, unitType, "DeviceAllow")
			if e == nil {
				if rv := reflect.ValueOf(devAllow.Value.Value()); rv.Kind() == reflect.Slice && rv.Len() == 0 {
					needsFreeze = false
					needsThaw = false
					return
				}
			}
		}
	}

	needsFreeze = true
	needsThaw = true

	// Check the current freezer state.
	freezerState, err := m.GetFreezerState()
	if err != nil {
		return
	}
	if freezerState == configs.Frozen {
		// Already frozen, and should stay frozen.
		needsFreeze = false
		needsThaw = false
	}

	if r.Freezer == configs.Frozen {
		// Will be frozen anyway -- no need to thaw.
		needsThaw = false
	}
	return
}

func (m *legacyManager) Set(r *configs.Resources) error {
	// If Paths are set, then we are just joining cgroups paths
	// and there is no need to set any values.
	if m.cgroups.Paths != nil {
		return nil
	}
	if r.Unified != nil {
		return cgroups.ErrV1NoUnified
	}
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
		if err := m.doFreeze(configs.Frozen); err != nil {
			// If freezer cgroup isn't supported, we just warn about it.
			logrus.Infof("freeze container before SetUnitProperties failed: %v", err)
		}
	}
	setErr := setUnitProperties(m.dbus, unitName, properties...)
	if needsThaw {
		if err := m.doFreeze(configs.Thawed); err != nil {
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

func (m *legacyManager) GetPaths() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths
}

func (m *legacyManager) GetCgroups() (*configs.Cgroup, error) {
	return m.cgroups, nil
}

func (m *legacyManager) GetFreezerState() (configs.FreezerState, error) {
	path, ok := m.paths["freezer"]
	if !ok {
		return configs.Undefined, nil
	}
	freezer := &fs.FreezerGroup{}
	return freezer.GetState(path)
}

func (m *legacyManager) Exists() bool {
	return cgroups.PathExists(m.Path("devices"))
}

func (m *legacyManager) OOMKillCount() (uint64, error) {
	return fs.OOMKillCount(m.Path("memory"))
}
