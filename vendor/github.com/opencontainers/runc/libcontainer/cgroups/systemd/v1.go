// +build linux

package systemd

import (
	"errors"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/sirupsen/logrus"
)

type legacyManager struct {
	mu      sync.Mutex
	cgroups *configs.Cgroup
	paths   map[string]string
}

func NewLegacyManager(cg *configs.Cgroup, paths map[string]string) cgroups.Manager {
	return &legacyManager{
		cgroups: cg,
		paths:   paths,
	}
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// Returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
	// Set the cgroup represented by cgroup.
	Set(path string, cgroup *configs.Cgroup) error
}

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

var legacySubsystems = subsystemSet{
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

func genV1ResourcesProperties(c *configs.Cgroup) ([]systemdDbus.Property, error) {
	var properties []systemdDbus.Property

	deviceProperties, err := generateDeviceProperties(c.Resources.Devices)
	if err != nil {
		return nil, err
	}
	properties = append(properties, deviceProperties...)

	if c.Resources.Memory != 0 {
		properties = append(properties,
			newProp("MemoryLimit", uint64(c.Resources.Memory)))
	}

	if c.Resources.CpuShares != 0 {
		properties = append(properties,
			newProp("CPUShares", c.Resources.CpuShares))
	}

	// cpu.cfs_quota_us and cpu.cfs_period_us are controlled by systemd.
	if c.Resources.CpuQuota != 0 && c.Resources.CpuPeriod != 0 {
		// corresponds to USEC_INFINITY in systemd
		// if USEC_INFINITY is provided, CPUQuota is left unbound by systemd
		// always setting a property value ensures we can apply a quota and remove it later
		cpuQuotaPerSecUSec := uint64(math.MaxUint64)
		if c.Resources.CpuQuota > 0 {
			// systemd converts CPUQuotaPerSecUSec (microseconds per CPU second) to CPUQuota
			// (integer percentage of CPU) internally.  This means that if a fractional percent of
			// CPU is indicated by Resources.CpuQuota, we need to round up to the nearest
			// 10ms (1% of a second) such that child cgroups can set the cpu.cfs_quota_us they expect.
			cpuQuotaPerSecUSec = uint64(c.Resources.CpuQuota*1000000) / c.Resources.CpuPeriod
			if cpuQuotaPerSecUSec%10000 != 0 {
				cpuQuotaPerSecUSec = ((cpuQuotaPerSecUSec / 10000) + 1) * 10000
			}
		}
		properties = append(properties,
			newProp("CPUQuotaPerSecUSec", cpuQuotaPerSecUSec))
	}

	if c.Resources.BlkioWeight != 0 {
		properties = append(properties,
			newProp("BlockIOWeight", uint64(c.Resources.BlkioWeight)))
	}

	if c.Resources.PidsLimit > 0 || c.Resources.PidsLimit == -1 {
		properties = append(properties,
			newProp("TasksAccounting", true),
			newProp("TasksMax", uint64(c.Resources.PidsLimit)))
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

	if c.Paths != nil {
		paths := make(map[string]string)
		for name, path := range c.Paths {
			_, err := getSubsystemPath(m.cgroups, name)
			if err != nil {
				// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
				if cgroups.IsNotFound(err) {
					continue
				}
				return err
			}
			paths[name] = path
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
		newProp("BlockIOAccounting", true))

	// Assume DefaultDependencies= will always work (the check for it was previously broken.)
	properties = append(properties,
		newProp("DefaultDependencies", false))

	resourcesProperties, err := genV1ResourcesProperties(c)
	if err != nil {
		return err
	}
	properties = append(properties, resourcesProperties...)
	properties = append(properties, c.SystemdProps...)

	// We have to set kernel memory here, as we can't change it once
	// processes have been attached to the cgroup.
	if c.Resources.KernelMemory != 0 {
		if err := enableKmem(c); err != nil {
			return err
		}
	}

	dbusConnection, err := getDbusConnection(false)
	if err != nil {
		return err
	}
	if err := startUnit(dbusConnection, unitName, properties); err != nil {
		return err
	}

	if err := joinCgroups(c, pid); err != nil {
		return err
	}

	paths := make(map[string]string)
	for _, s := range legacySubsystems {
		subsystemPath, err := getSubsystemPath(m.cgroups, s.Name())
		if err != nil {
			// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
			if cgroups.IsNotFound(err) {
				continue
			}
			return err
		}
		paths[s.Name()] = subsystemPath
	}
	m.paths = paths
	return nil
}

func (m *legacyManager) Destroy() error {
	if m.cgroups.Paths != nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	dbusConnection, err := getDbusConnection(false)
	if err != nil {
		return err
	}
	unitName := getUnitName(m.cgroups)
	if err := stopUnit(dbusConnection, unitName); err != nil {
		return err
	}
	m.paths = make(map[string]string)
	return nil
}

func (m *legacyManager) Path(subsys string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths[subsys]
}

func join(c *configs.Cgroup, subsystem string, pid int) (string, error) {
	path, err := getSubsystemPath(c, subsystem)
	if err != nil {
		return "", err
	}

	if err := os.MkdirAll(path, 0755); err != nil {
		return "", err
	}
	if err := cgroups.WriteCgroupProc(path, pid); err != nil {
		return "", err
	}
	return path, nil
}

func joinCgroups(c *configs.Cgroup, pid int) error {
	for _, sys := range legacySubsystems {
		name := sys.Name()
		switch name {
		case "name=systemd":
			// let systemd handle this
		case "cpuset":
			path, err := getSubsystemPath(c, name)
			if err != nil && !cgroups.IsNotFound(err) {
				return err
			}
			s := &fs.CpusetGroup{}
			if err := s.ApplyDir(path, c, pid); err != nil {
				return err
			}
		default:
			_, err := join(c, name, pid)
			if err != nil {
				// Even if it's `not found` error, we'll return err
				// because devices cgroup is hard requirement for
				// container security.
				if name == "devices" {
					return err
				}
				// For other subsystems, omit the `not found` error
				// because they are optional.
				if !cgroups.IsNotFound(err) {
					return err
				}
			}
		}
	}

	return nil
}

func getSubsystemPath(c *configs.Cgroup, subsystem string) (string, error) {
	mountpoint, err := cgroups.FindCgroupMountpoint(c.Path, subsystem)
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
	path, err := getSubsystemPath(m.cgroups, "freezer")
	if err != nil {
		return err
	}
	prevState := m.cgroups.Resources.Freezer
	m.cgroups.Resources.Freezer = state
	freezer, err := legacySubsystems.Get("freezer")
	if err != nil {
		return err
	}
	err = freezer.Set(path, m.cgroups)
	if err != nil {
		m.cgroups.Resources.Freezer = prevState
		return err
	}
	return nil
}

func (m *legacyManager) GetPids() ([]int, error) {
	path, err := getSubsystemPath(m.cgroups, "devices")
	if err != nil {
		return nil, err
	}
	return cgroups.GetPids(path)
}

func (m *legacyManager) GetAllPids() ([]int, error) {
	path, err := getSubsystemPath(m.cgroups, "devices")
	if err != nil {
		return nil, err
	}
	return cgroups.GetAllPids(path)
}

func (m *legacyManager) GetStats() (*cgroups.Stats, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	stats := cgroups.NewStats()
	for name, path := range m.paths {
		sys, err := legacySubsystems.Get(name)
		if err == errSubsystemDoesNotExist || !cgroups.PathExists(path) {
			continue
		}
		if err := sys.GetStats(path, stats); err != nil {
			return nil, err
		}
	}

	return stats, nil
}

func (m *legacyManager) Set(container *configs.Config) error {
	// If Paths are set, then we are just joining cgroups paths
	// and there is no need to set any values.
	if m.cgroups.Paths != nil {
		return nil
	}
	properties, err := genV1ResourcesProperties(container.Cgroups)
	if err != nil {
		return err
	}

	// Figure out the current freezer state, so we can revert to it after we
	// temporarily freeze the container.
	targetFreezerState, err := m.GetFreezerState()
	if err != nil {
		return err
	}
	if targetFreezerState == configs.Undefined {
		targetFreezerState = configs.Thawed
	}

	// We have to freeze the container while systemd sets the cgroup settings.
	// The reason for this is that systemd's application of DeviceAllow rules
	// is done disruptively, resulting in spurrious errors to common devices
	// (unlike our fs driver, they will happily write deny-all rules to running
	// containers). So we freeze the container to avoid them hitting the cgroup
	// error. But if the freezer cgroup isn't supported, we just warn about it.
	if err := m.Freeze(configs.Frozen); err != nil {
		logrus.Infof("freeze container before SetUnitProperties failed: %v", err)
	}

	dbusConnection, err := getDbusConnection(false)
	if err != nil {
		_ = m.Freeze(targetFreezerState)
		return err
	}
	if err := dbusConnection.SetUnitProperties(getUnitName(container.Cgroups), true, properties...); err != nil {
		_ = m.Freeze(targetFreezerState)
		return err
	}

	// Reset freezer state before we apply the configuration, to avoid clashing
	// with the freezer setting in the configuration.
	_ = m.Freeze(targetFreezerState)

	for _, sys := range legacySubsystems {
		// Get the subsystem path, but don't error out for not found cgroups.
		path, err := getSubsystemPath(container.Cgroups, sys.Name())
		if err != nil && !cgroups.IsNotFound(err) {
			return err
		}
		if err := sys.Set(path, container.Cgroups); err != nil {
			return err
		}
	}

	return nil
}

func enableKmem(c *configs.Cgroup) error {
	path, err := getSubsystemPath(c, "memory")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}

	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}
	// do not try to enable the kernel memory if we already have
	// tasks in the cgroup.
	content, err := ioutil.ReadFile(filepath.Join(path, "tasks"))
	if err != nil {
		return err
	}
	if len(content) > 0 {
		return nil
	}
	return fs.EnableKernelMemoryAccounting(path)
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
	path, err := getSubsystemPath(m.cgroups, "freezer")
	if err != nil && !cgroups.IsNotFound(err) {
		return configs.Undefined, err
	}
	freezer, err := legacySubsystems.Get("freezer")
	if err != nil {
		return configs.Undefined, err
	}
	return freezer.(*fs.FreezerGroup).GetState(path)
}
