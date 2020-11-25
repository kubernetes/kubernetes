// +build linux

package systemd

import (
	"errors"
	"io/ioutil"
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

func genV1ResourcesProperties(c *configs.Cgroup, conn *systemdDbus.Conn) ([]systemdDbus.Property, error) {
	var properties []systemdDbus.Property
	r := c.Resources

	deviceProperties, err := generateDeviceProperties(r.Devices)
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

	addCpuQuota(conn, &properties, r.CpuQuota, r.CpuPeriod)

	if r.BlkioWeight != 0 {
		properties = append(properties,
			newProp("BlockIOWeight", uint64(r.BlkioWeight)))
	}

	if r.PidsLimit > 0 || r.PidsLimit == -1 {
		properties = append(properties,
			newProp("TasksAccounting", true),
			newProp("TasksMax", uint64(r.PidsLimit)))
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

	m.mu.Lock()
	defer m.mu.Unlock()
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

	dbusConnection, err := getDbusConnection(false)
	if err != nil {
		return err
	}
	resourcesProperties, err := genV1ResourcesProperties(c, dbusConnection)
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

	err = stopUnit(dbusConnection, unitName)
	// Both on success and on error, cleanup all the cgroups we are aware of.
	// Some of them were created directly by Apply() and are not managed by systemd.
	if err := cgroups.RemovePaths(m.paths); err != nil {
		return err
	}
	if err != nil {
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
	freezer := &fs.FreezerGroup{}
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

func (m *legacyManager) Set(container *configs.Config) error {
	// If Paths are set, then we are just joining cgroups paths
	// and there is no need to set any values.
	if m.cgroups.Paths != nil {
		return nil
	}
	dbusConnection, err := getDbusConnection(false)
	if err != nil {
		return err
	}
	properties, err := genV1ResourcesProperties(container.Cgroups, dbusConnection)
	if err != nil {
		return err
	}

	// We have to freeze the container while systemd sets the cgroup settings.
	// The reason for this is that systemd's application of DeviceAllow rules
	// is done disruptively, resulting in spurrious errors to common devices
	// (unlike our fs driver, they will happily write deny-all rules to running
	// containers). So we freeze the container to avoid them hitting the cgroup
	// error. But if the freezer cgroup isn't supported, we just warn about it.
	targetFreezerState := configs.Undefined
	if !m.cgroups.SkipDevices {
		// Figure out the current freezer state, so we can revert to it after we
		// temporarily freeze the container.
		targetFreezerState, err = m.GetFreezerState()
		if err != nil {
			return err
		}
		if targetFreezerState == configs.Undefined {
			targetFreezerState = configs.Thawed
		}

		if err := m.Freeze(configs.Frozen); err != nil {
			logrus.Infof("freeze container before SetUnitProperties failed: %v", err)
		}
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
	freezer := &fs.FreezerGroup{}
	return freezer.GetState(path)
}

func (m *legacyManager) Exists() bool {
	return cgroups.PathExists(m.Path("devices"))
}
