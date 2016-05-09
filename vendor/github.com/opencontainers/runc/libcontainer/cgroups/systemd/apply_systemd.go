// +build linux

package systemd

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	systemdDbus "github.com/coreos/go-systemd/dbus"
	systemdUtil "github.com/coreos/go-systemd/util"
	"github.com/godbus/dbus"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type Manager struct {
	mu      sync.Mutex
	Cgroups *configs.Cgroup
	Paths   map[string]string
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

var subsystems = subsystemSet{
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

const (
	testScopeWait = 4
)

var (
	connLock                        sync.Mutex
	theConn                         *systemdDbus.Conn
	hasStartTransientUnit           bool
	hasTransientDefaultDependencies bool
)

func newProp(name string, units interface{}) systemdDbus.Property {
	return systemdDbus.Property{
		Name:  name,
		Value: dbus.MakeVariant(units),
	}
}

func UseSystemd() bool {
	if !systemdUtil.IsRunningSystemd() {
		return false
	}

	connLock.Lock()
	defer connLock.Unlock()

	if theConn == nil {
		var err error
		theConn, err = systemdDbus.New()
		if err != nil {
			return false
		}

		// Assume we have StartTransientUnit
		hasStartTransientUnit = true

		// But if we get UnknownMethod error we don't
		if _, err := theConn.StartTransientUnit("test.scope", "invalid", nil, nil); err != nil {
			if dbusError, ok := err.(dbus.Error); ok {
				if dbusError.Name == "org.freedesktop.DBus.Error.UnknownMethod" {
					hasStartTransientUnit = false
					return hasStartTransientUnit
				}
			}
		}

		// Ensure the scope name we use doesn't exist. Use the Pid to
		// avoid collisions between multiple libcontainer users on a
		// single host.
		scope := fmt.Sprintf("libcontainer-%d-systemd-test-default-dependencies.scope", os.Getpid())
		testScopeExists := true
		for i := 0; i <= testScopeWait; i++ {
			if _, err := theConn.StopUnit(scope, "replace", nil); err != nil {
				if dbusError, ok := err.(dbus.Error); ok {
					if strings.Contains(dbusError.Name, "org.freedesktop.systemd1.NoSuchUnit") {
						testScopeExists = false
						break
					}
				}
			}
			time.Sleep(time.Millisecond)
		}

		// Bail out if we can't kill this scope without testing for DefaultDependencies
		if testScopeExists {
			return hasStartTransientUnit
		}

		// Assume StartTransientUnit on a scope allows DefaultDependencies
		hasTransientDefaultDependencies = true
		ddf := newProp("DefaultDependencies", false)
		if _, err := theConn.StartTransientUnit(scope, "replace", []systemdDbus.Property{ddf}, nil); err != nil {
			if dbusError, ok := err.(dbus.Error); ok {
				if strings.Contains(dbusError.Name, "org.freedesktop.DBus.Error.PropertyReadOnly") {
					hasTransientDefaultDependencies = false
				}
			}
		}

		// Not critical because of the stop unit logic above.
		theConn.StopUnit(scope, "replace", nil)
	}
	return hasStartTransientUnit
}

func getIfaceForUnit(unitName string) string {
	if strings.HasSuffix(unitName, ".scope") {
		return "Scope"
	}
	if strings.HasSuffix(unitName, ".service") {
		return "Service"
	}
	return "Unit"
}

func (m *Manager) Apply(pid int) error {
	var (
		c          = m.Cgroups
		unitName   = getUnitName(c)
		slice      = "system.slice"
		properties []systemdDbus.Property
	)

	if c.Paths != nil {
		paths := make(map[string]string)
		for name, path := range c.Paths {
			_, err := getSubsystemPath(m.Cgroups, name)
			if err != nil {
				// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
				if cgroups.IsNotFound(err) {
					continue
				}
				return err
			}
			paths[name] = path
		}
		m.Paths = paths
		return cgroups.EnterPid(m.Paths, pid)
	}

	if c.Parent != "" {
		slice = c.Parent
	}

	properties = append(properties,
		systemdDbus.PropSlice(slice),
		systemdDbus.PropDescription("docker container "+c.Name),
		newProp("PIDs", []uint32{uint32(pid)}),
	)

	// Always enable accounting, this gets us the same behaviour as the fs implementation,
	// plus the kernel has some problems with joining the memory cgroup at a later time.
	properties = append(properties,
		newProp("MemoryAccounting", true),
		newProp("CPUAccounting", true),
		newProp("BlockIOAccounting", true))

	if hasTransientDefaultDependencies {
		properties = append(properties,
			newProp("DefaultDependencies", false))
	}

	if c.Resources.Memory != 0 {
		properties = append(properties,
			newProp("MemoryLimit", uint64(c.Resources.Memory)))
	}

	if c.Resources.CpuShares != 0 {
		properties = append(properties,
			newProp("CPUShares", uint64(c.Resources.CpuShares)))
	}

	if c.Resources.BlkioWeight != 0 {
		properties = append(properties,
			newProp("BlockIOWeight", uint64(c.Resources.BlkioWeight)))
	}

	// We need to set kernel memory before processes join cgroup because
	// kmem.limit_in_bytes can only be set when the cgroup is empty.
	// And swap memory limit needs to be set after memory limit, only
	// memory limit is handled by systemd, so it's kind of ugly here.
	if c.Resources.KernelMemory > 0 {
		if err := setKernelMemory(c); err != nil {
			return err
		}
	}

	if _, err := theConn.StartTransientUnit(unitName, "replace", properties, nil); err != nil {
		return err
	}

	if err := joinDevices(c, pid); err != nil {
		return err
	}

	// TODO: CpuQuota and CpuPeriod not available in systemd
	// we need to manually join the cpu.cfs_quota_us and cpu.cfs_period_us
	if err := joinCpu(c, pid); err != nil {
		return err
	}

	// TODO: MemoryReservation and MemorySwap not available in systemd
	if err := joinMemory(c, pid); err != nil {
		return err
	}

	// we need to manually join the freezer, net_cls, net_prio, pids and cpuset cgroup in systemd
	// because it does not currently support it via the dbus api.
	if err := joinFreezer(c, pid); err != nil {
		return err
	}

	if err := joinNetPrio(c, pid); err != nil {
		return err
	}
	if err := joinNetCls(c, pid); err != nil {
		return err
	}

	if err := joinPids(c, pid); err != nil {
		return err
	}

	if err := joinCpuset(c, pid); err != nil {
		return err
	}

	if err := joinHugetlb(c, pid); err != nil {
		return err
	}

	if err := joinPerfEvent(c, pid); err != nil {
		return err
	}
	// FIXME: Systemd does have `BlockIODeviceWeight` property, but we got problem
	// using that (at least on systemd 208, see https://github.com/opencontainers/runc/libcontainer/pull/354),
	// so use fs work around for now.
	if err := joinBlkio(c, pid); err != nil {
		return err
	}

	paths := make(map[string]string)
	for _, s := range subsystems {
		subsystemPath, err := getSubsystemPath(m.Cgroups, s.Name())
		if err != nil {
			// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
			if cgroups.IsNotFound(err) {
				continue
			}
			return err
		}
		paths[s.Name()] = subsystemPath
	}
	m.Paths = paths
	return nil
}

func (m *Manager) Destroy() error {
	if m.Cgroups.Paths != nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	theConn.StopUnit(getUnitName(m.Cgroups), "replace", nil)
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

func writeFile(dir, file, data string) error {
	// Normally dir should not be empty, one case is that cgroup subsystem
	// is not mounted, we will get empty dir, and we want it fail here.
	if dir == "" {
		return fmt.Errorf("no such directory for %s.", file)
	}
	return ioutil.WriteFile(filepath.Join(dir, file), []byte(data), 0700)
}

func join(c *configs.Cgroup, subsystem string, pid int) (string, error) {
	path, err := getSubsystemPath(c, subsystem)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(path, 0755); err != nil {
		return "", err
	}
	if err := writeFile(path, "cgroup.procs", strconv.Itoa(pid)); err != nil {
		return "", err
	}

	return path, nil
}

func joinCpu(c *configs.Cgroup, pid int) error {
	_, err := join(c, "cpu", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func joinFreezer(c *configs.Cgroup, pid int) error {
	_, err := join(c, "freezer", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func joinNetPrio(c *configs.Cgroup, pid int) error {
	_, err := join(c, "net_prio", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func joinNetCls(c *configs.Cgroup, pid int) error {
	_, err := join(c, "net_cls", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func joinPids(c *configs.Cgroup, pid int) error {
	_, err := join(c, "pids", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

// systemd represents slice heirarchy using `-`, so we need to follow suit when
// generating the path of slice. Essentially, test-a-b.slice becomes
// test.slice/test-a.slice/test-a-b.slice.
func expandSlice(slice string) (string, error) {
	suffix := ".slice"
	sliceName := strings.TrimSuffix(slice, suffix)

	var path, prefix string
	for _, component := range strings.Split(sliceName, "-") {
		// test--a.slice isn't permitted, nor is -test.slice.
		if component == "" {
			return "", fmt.Errorf("invalid slice name: %s", slice)
		}

		// Append the component to the path and to the prefix.
		path += prefix + component + suffix + "/"
		prefix += component + "-"
	}

	return path, nil
}

func getSubsystemPath(c *configs.Cgroup, subsystem string) (string, error) {
	mountpoint, err := cgroups.FindCgroupMountpoint(subsystem)
	if err != nil {
		return "", err
	}

	initPath, err := cgroups.GetInitCgroupDir(subsystem)
	if err != nil {
		return "", err
	}

	slice := "system.slice"
	if c.Parent != "" {
		slice = c.Parent
	}

	slice, err = expandSlice(slice)
	if err != nil {
		return "", err
	}

	return filepath.Join(mountpoint, initPath, slice, getUnitName(c)), nil
}

func (m *Manager) Freeze(state configs.FreezerState) error {
	path, err := getSubsystemPath(m.Cgroups, "freezer")
	if err != nil {
		return err
	}
	prevState := m.Cgroups.Resources.Freezer
	m.Cgroups.Resources.Freezer = state
	freezer, err := subsystems.Get("freezer")
	if err != nil {
		return err
	}
	err = freezer.Set(path, m.Cgroups)
	if err != nil {
		m.Cgroups.Resources.Freezer = prevState
		return err
	}
	return nil
}

func (m *Manager) GetPids() ([]int, error) {
	path, err := getSubsystemPath(m.Cgroups, "devices")
	if err != nil {
		return nil, err
	}
	return cgroups.GetPids(path)
}

func (m *Manager) GetAllPids() ([]int, error) {
	path, err := getSubsystemPath(m.Cgroups, "devices")
	if err != nil {
		return nil, err
	}
	return cgroups.GetAllPids(path)
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
	for _, sys := range subsystems {
		// Get the subsystem path, but don't error out for not found cgroups.
		path, err := getSubsystemPath(container.Cgroups, sys.Name())
		if err != nil && !cgroups.IsNotFound(err) {
			return err
		}

		if err := sys.Set(path, container.Cgroups); err != nil {
			return err
		}
	}

	if m.Paths["cpu"] != "" {
		if err := fs.CheckCpushares(m.Paths["cpu"], container.Cgroups.Resources.CpuShares); err != nil {
			return err
		}
	}
	return nil
}

func getUnitName(c *configs.Cgroup) string {
	return fmt.Sprintf("%s-%s.scope", c.ScopePrefix, c.Name)
}

// Atm we can't use the systemd device support because of two missing things:
// * Support for wildcards to allow mknod on any device
// * Support for wildcards to allow /dev/pts support
//
// The second is available in more recent systemd as "char-pts", but not in e.g. v208 which is
// in wide use. When both these are available we will be able to switch, but need to keep the old
// implementation for backwards compat.
//
// Note: we can't use systemd to set up the initial limits, and then change the cgroup
// because systemd will re-write the device settings if it needs to re-apply the cgroup context.
// This happens at least for v208 when any sibling unit is started.
func joinDevices(c *configs.Cgroup, pid int) error {
	_, err := join(c, "devices", pid)
	// Even if it's `not found` error, we'll return err because devices cgroup
	// is hard requirement for container security.
	if err != nil {
		return err
	}
	return nil
}

func setKernelMemory(c *configs.Cgroup) error {
	path, err := getSubsystemPath(c, "memory")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}

	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}

	// This doesn't get called by manager.Set, so we need to do it here.
	s := &fs.MemoryGroup{}
	return s.SetKernelMemory(path, c)
}

func joinMemory(c *configs.Cgroup, pid int) error {
	_, err := join(c, "memory", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

// systemd does not atm set up the cpuset controller, so we must manually
// join it. Additionally that is a very finicky controller where each
// level must have a full setup as the default for a new directory is "no cpus"
func joinCpuset(c *configs.Cgroup, pid int) error {
	path, err := getSubsystemPath(c, "cpuset")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}

	s := &fs.CpusetGroup{}

	return s.ApplyDir(path, c, pid)
}

// `BlockIODeviceWeight` property of systemd does not work properly, and systemd
// expects device path instead of major minor numbers, which is also confusing
// for users. So we use fs work around for now.
func joinBlkio(c *configs.Cgroup, pid int) error {
	_, err := join(c, "blkio", pid)
	if err != nil {
		return err
	}
	return nil
}

func joinHugetlb(c *configs.Cgroup, pid int) error {
	_, err := join(c, "hugetlb", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func joinPerfEvent(c *configs.Cgroup, pid int) error {
	_, err := join(c, "perf_event", pid)
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}
