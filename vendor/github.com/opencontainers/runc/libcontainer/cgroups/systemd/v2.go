// +build linux

package systemd

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs2"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

type unifiedManager struct {
	mu      sync.Mutex
	cgroups *configs.Cgroup
	// path is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope"
	path     string
	rootless bool
}

func NewUnifiedManager(config *configs.Cgroup, path string, rootless bool) cgroups.Manager {
	return &unifiedManager{
		cgroups:  config,
		path:     path,
		rootless: rootless,
	}
}

func genV2ResourcesProperties(c *configs.Cgroup, conn *systemdDbus.Conn) ([]systemdDbus.Property, error) {
	var properties []systemdDbus.Property
	r := c.Resources

	// NOTE: This is of questionable correctness because we insert our own
	//       devices eBPF program later. Two programs with identical rules
	//       aren't the end of the world, but it is a bit concerning. However
	//       it's unclear if systemd removes all eBPF programs attached when
	//       doing SetUnitProperties...
	deviceProperties, err := generateDeviceProperties(r.Devices)
	if err != nil {
		return nil, err
	}
	properties = append(properties, deviceProperties...)

	if r.Memory != 0 {
		properties = append(properties,
			newProp("MemoryMax", uint64(r.Memory)))
	}
	if r.MemoryReservation != 0 {
		properties = append(properties,
			newProp("MemoryLow", uint64(r.MemoryReservation)))
	}

	swap, err := cgroups.ConvertMemorySwapToCgroupV2Value(r.MemorySwap, r.Memory)
	if err != nil {
		return nil, err
	}
	if swap != 0 {
		properties = append(properties,
			newProp("MemorySwapMax", uint64(swap)))
	}

	if r.CpuWeight != 0 {
		properties = append(properties,
			newProp("CPUWeight", r.CpuWeight))
	}

	addCpuQuota(conn, &properties, r.CpuQuota, r.CpuPeriod)

	if r.PidsLimit > 0 || r.PidsLimit == -1 {
		properties = append(properties,
			newProp("TasksAccounting", true),
			newProp("TasksMax", uint64(r.PidsLimit)))
	}

	// ignore r.KernelMemory

	return properties, nil
}

func (m *unifiedManager) Apply(pid int) error {
	var (
		c          = m.cgroups
		unitName   = getUnitName(c)
		properties []systemdDbus.Property
	)

	if c.Paths != nil {
		return cgroups.WriteCgroupProc(m.path, pid)
	}

	slice := "system.slice"
	if m.rootless {
		slice = "user.slice"
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
		newProp("IOAccounting", true))

	// Assume DefaultDependencies= will always work (the check for it was previously broken.)
	properties = append(properties,
		newProp("DefaultDependencies", false))

	dbusConnection, err := getDbusConnection(m.rootless)
	if err != nil {
		return err
	}
	resourcesProperties, err := genV2ResourcesProperties(c, dbusConnection)
	if err != nil {
		return err
	}
	properties = append(properties, resourcesProperties...)
	properties = append(properties, c.SystemdProps...)

	if err := startUnit(dbusConnection, unitName, properties); err != nil {
		return errors.Wrapf(err, "error while starting unit %q with properties %+v", unitName, properties)
	}

	if err = m.initPath(); err != nil {
		return err
	}
	if err := fs2.CreateCgroupPath(m.path, m.cgroups); err != nil {
		return err
	}
	return nil
}

func (m *unifiedManager) Destroy() error {
	if m.cgroups.Paths != nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	dbusConnection, err := getDbusConnection(m.rootless)
	if err != nil {
		return err
	}
	unitName := getUnitName(m.cgroups)
	if err := stopUnit(dbusConnection, unitName); err != nil {
		return err
	}

	// XXX this is probably not needed, systemd should handle it
	err = os.Remove(m.path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}

func (m *unifiedManager) Path(_ string) string {
	return m.path
}

// getSliceFull value is used in initPath.
// The value is incompatible with systemdDbus.PropSlice.
func (m *unifiedManager) getSliceFull() (string, error) {
	c := m.cgroups
	slice := "system.slice"
	if m.rootless {
		slice = "user.slice"
	}
	if c.Parent != "" {
		var err error
		slice, err = ExpandSlice(c.Parent)
		if err != nil {
			return "", err
		}
	}

	if m.rootless {
		dbusConnection, err := getDbusConnection(m.rootless)
		if err != nil {
			return "", err
		}
		// managerCGQuoted is typically "/user.slice/user-${uid}.slice/user@${uid}.service" including the quote symbols
		managerCGQuoted, err := dbusConnection.GetManagerProperty("ControlGroup")
		if err != nil {
			return "", err
		}
		managerCG, err := strconv.Unquote(managerCGQuoted)
		if err != nil {
			return "", err
		}
		slice = filepath.Join(managerCG, slice)
	}

	// an example of the final slice in rootless: "/user.slice/user-1001.slice/user@1001.service/user.slice"
	// NOTE: systemdDbus.PropSlice requires the "/user.slice/user-1001.slice/user@1001.service/" prefix NOT to be specified.
	return slice, nil
}

func (m *unifiedManager) initPath() error {
	if m.path != "" {
		return nil
	}

	sliceFull, err := m.getSliceFull()
	if err != nil {
		return err
	}

	c := m.cgroups
	path := filepath.Join(sliceFull, getUnitName(c))
	path, err = securejoin.SecureJoin(fs2.UnifiedMountpoint, path)
	if err != nil {
		return err
	}

	// an example of the final path in rootless:
	// "/sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/user.slice/libpod-132ff0d72245e6f13a3bbc6cdc5376886897b60ac59eaa8dea1df7ab959cbf1c.scope"
	m.path = path

	return nil
}

func (m *unifiedManager) fsManager() (cgroups.Manager, error) {
	if err := m.initPath(); err != nil {
		return nil, err
	}
	return fs2.NewManager(m.cgroups, m.path, m.rootless)
}

func (m *unifiedManager) Freeze(state configs.FreezerState) error {
	fsMgr, err := m.fsManager()
	if err != nil {
		return err
	}
	return fsMgr.Freeze(state)
}

func (m *unifiedManager) GetPids() ([]int, error) {
	if err := m.initPath(); err != nil {
		return nil, err
	}
	return cgroups.GetPids(m.path)
}

func (m *unifiedManager) GetAllPids() ([]int, error) {
	if err := m.initPath(); err != nil {
		return nil, err
	}
	return cgroups.GetAllPids(m.path)
}

func (m *unifiedManager) GetStats() (*cgroups.Stats, error) {
	fsMgr, err := m.fsManager()
	if err != nil {
		return nil, err
	}
	return fsMgr.GetStats()
}

func (m *unifiedManager) Set(container *configs.Config) error {
	dbusConnection, err := getDbusConnection(m.rootless)
	if err != nil {
		return err
	}
	properties, err := genV2ResourcesProperties(m.cgroups, dbusConnection)
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

	if err := dbusConnection.SetUnitProperties(getUnitName(m.cgroups), true, properties...); err != nil {
		_ = m.Freeze(targetFreezerState)
		return errors.Wrap(err, "error while setting unit properties")
	}

	// Reset freezer state before we apply the configuration, to avoid clashing
	// with the freezer setting in the configuration.
	_ = m.Freeze(targetFreezerState)

	fsMgr, err := m.fsManager()
	if err != nil {
		return err
	}
	return fsMgr.Set(container)
}

func (m *unifiedManager) GetPaths() map[string]string {
	paths := make(map[string]string, 1)
	paths[""] = m.path
	return paths
}

func (m *unifiedManager) GetCgroups() (*configs.Cgroup, error) {
	return m.cgroups, nil
}

func (m *unifiedManager) GetFreezerState() (configs.FreezerState, error) {
	fsMgr, err := m.fsManager()
	if err != nil {
		return configs.Undefined, err
	}
	return fsMgr.GetFreezerState()
}

func (m *unifiedManager) Exists() bool {
	return cgroups.PathExists(m.path)
}
