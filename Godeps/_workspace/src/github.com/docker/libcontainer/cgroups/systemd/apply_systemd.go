// +build linux

package systemd

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	systemd "github.com/coreos/go-systemd/dbus"
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/cgroups/fs"
	"github.com/docker/libcontainer/configs"
	"github.com/godbus/dbus"
)

type Manager struct {
	Cgroups *configs.Cgroup
	Paths   map[string]string
}

type subsystem interface {
	// Returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
	// Set the cgroup represented by cgroup.
	Set(path string, cgroup *configs.Cgroup) error
}

var subsystems = map[string]subsystem{
	"devices":    &fs.DevicesGroup{},
	"memory":     &fs.MemoryGroup{},
	"cpu":        &fs.CpuGroup{},
	"cpuset":     &fs.CpusetGroup{},
	"cpuacct":    &fs.CpuacctGroup{},
	"blkio":      &fs.BlkioGroup{},
	"hugetlb":    &fs.HugetlbGroup{},
	"perf_event": &fs.PerfEventGroup{},
	"freezer":    &fs.FreezerGroup{},
}

const (
	testScopeWait = 4
)

var (
	connLock                        sync.Mutex
	theConn                         *systemd.Conn
	hasStartTransientUnit           bool
	hasTransientDefaultDependencies bool
)

func newProp(name string, units interface{}) systemd.Property {
	return systemd.Property{
		Name:  name,
		Value: dbus.MakeVariant(units),
	}
}

func UseSystemd() bool {
	s, err := os.Stat("/run/systemd/system")
	if err != nil || !s.IsDir() {
		return false
	}

	connLock.Lock()
	defer connLock.Unlock()

	if theConn == nil {
		var err error
		theConn, err = systemd.New()
		if err != nil {
			return false
		}

		// Assume we have StartTransientUnit
		hasStartTransientUnit = true

		// But if we get UnknownMethod error we don't
		if _, err := theConn.StartTransientUnit("test.scope", "invalid"); err != nil {
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
			if _, err := theConn.StopUnit(scope, "replace"); err != nil {
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
		if _, err := theConn.StartTransientUnit(scope, "replace", ddf); err != nil {
			if dbusError, ok := err.(dbus.Error); ok {
				if strings.Contains(dbusError.Name, "org.freedesktop.DBus.Error.PropertyReadOnly") {
					hasTransientDefaultDependencies = false
				}
			}
		}

		// Not critical because of the stop unit logic above.
		theConn.StopUnit(scope, "replace")
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
		properties []systemd.Property
	)

	if c.Slice != "" {
		slice = c.Slice
	}

	properties = append(properties,
		systemd.PropSlice(slice),
		systemd.PropDescription("docker container "+c.Name),
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

	if c.Memory != 0 {
		properties = append(properties,
			newProp("MemoryLimit", uint64(c.Memory)))
	}
	// TODO: MemoryReservation and MemorySwap not available in systemd

	if c.CpuShares != 0 {
		properties = append(properties,
			newProp("CPUShares", uint64(c.CpuShares)))
	}

	if c.BlkioWeight != 0 {
		properties = append(properties,
			newProp("BlockIOWeight", uint64(c.BlkioWeight)))
	}

	if _, err := theConn.StartTransientUnit(unitName, "replace", properties...); err != nil {
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

	// -1 disables memorySwap
	if c.MemorySwap >= 0 && c.Memory != 0 {
		if err := joinMemory(c, pid); err != nil {
			return err
		}

	}

	// we need to manually join the freezer and cpuset cgroup in systemd
	// because it does not currently support it via the dbus api.
	if err := joinFreezer(c, pid); err != nil {
		return err
	}

	if err := joinCpuset(c, pid); err != nil {
		return err
	}

	// FIXME: Systemd does have `BlockIODeviceWeight` property, but we got problem
	// using that (at least on systemd 208, see https://github.com/docker/libcontainer/pull/354),
	// so use fs work around for now.
	if err := joinBlkio(c, pid); err != nil {
		return err
	}

	paths := make(map[string]string)
	for sysname := range subsystems {
		subsystemPath, err := getSubsystemPath(m.Cgroups, sysname)
		if err != nil {
			// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
			if cgroups.IsNotFound(err) {
				continue
			}
			return err
		}
		paths[sysname] = subsystemPath
	}
	m.Paths = paths

	if paths["cpu"] != "" {
		if err := fs.CheckCpushares(paths["cpu"], c.CpuShares); err != nil {
			return err
		}
	}

	return nil
}

func (m *Manager) Destroy() error {
	return cgroups.RemovePaths(m.Paths)
}

func (m *Manager) GetPaths() map[string]string {
	return m.Paths
}

func writeFile(dir, file, data string) error {
	return ioutil.WriteFile(filepath.Join(dir, file), []byte(data), 0700)
}

func join(c *configs.Cgroup, subsystem string, pid int) (string, error) {
	path, err := getSubsystemPath(c, subsystem)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(path, 0755); err != nil && !os.IsExist(err) {
		return "", err
	}
	if err := writeFile(path, "cgroup.procs", strconv.Itoa(pid)); err != nil {
		return "", err
	}

	return path, nil
}

func joinCpu(c *configs.Cgroup, pid int) error {
	path, err := getSubsystemPath(c, "cpu")
	if err != nil {
		return err
	}
	if c.CpuQuota != 0 {
		if err = ioutil.WriteFile(filepath.Join(path, "cpu.cfs_quota_us"), []byte(strconv.FormatInt(c.CpuQuota, 10)), 0700); err != nil {
			return err
		}
	}
	if c.CpuPeriod != 0 {
		if err = ioutil.WriteFile(filepath.Join(path, "cpu.cfs_period_us"), []byte(strconv.FormatInt(c.CpuPeriod, 10)), 0700); err != nil {
			return err
		}
	}
	return nil
}

func joinFreezer(c *configs.Cgroup, pid int) error {
	if _, err := join(c, "freezer", pid); err != nil {
		return err
	}

	return nil
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
	if c.Slice != "" {
		slice = c.Slice
	}

	return filepath.Join(mountpoint, initPath, slice, getUnitName(c)), nil
}

func (m *Manager) Freeze(state configs.FreezerState) error {
	path, err := getSubsystemPath(m.Cgroups, "freezer")
	if err != nil {
		return err
	}

	prevState := m.Cgroups.Freezer
	m.Cgroups.Freezer = state

	freezer := subsystems["freezer"]
	err = freezer.Set(path, m.Cgroups)
	if err != nil {
		m.Cgroups.Freezer = prevState
		return err
	}

	return nil
}

func (m *Manager) GetPids() ([]int, error) {
	path, err := getSubsystemPath(m.Cgroups, "cpu")
	if err != nil {
		return nil, err
	}

	return cgroups.ReadProcsFile(path)
}

func (m *Manager) GetStats() (*cgroups.Stats, error) {
	stats := cgroups.NewStats()
	for name, path := range m.Paths {
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

func (m *Manager) Set(container *configs.Config) error {
	for name, path := range m.Paths {
		sys, ok := subsystems[name]
		if !ok || !cgroups.PathExists(path) {
			continue
		}
		if err := sys.Set(path, container.Cgroups); err != nil {
			return err
		}
	}

	return nil
}

func getUnitName(c *configs.Cgroup) string {
	return fmt.Sprintf("%s-%s.scope", c.Parent, c.Name)
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
	path, err := join(c, "devices", pid)
	if err != nil {
		return err
	}

	devices := subsystems["devices"]
	return devices.Set(path, c)
}

func joinMemory(c *configs.Cgroup, pid int) error {
	memorySwap := c.MemorySwap

	if memorySwap == 0 {
		// By default, MemorySwap is set to twice the size of RAM.
		memorySwap = c.Memory * 2
	}

	path, err := getSubsystemPath(c, "memory")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filepath.Join(path, "memory.memsw.limit_in_bytes"), []byte(strconv.FormatInt(memorySwap, 10)), 0700)
}

// systemd does not atm set up the cpuset controller, so we must manually
// join it. Additionally that is a very finicky controller where each
// level must have a full setup as the default for a new directory is "no cpus"
func joinCpuset(c *configs.Cgroup, pid int) error {
	path, err := getSubsystemPath(c, "cpuset")
	if err != nil {
		return err
	}

	s := &fs.CpusetGroup{}

	return s.ApplyDir(path, c, pid)
}

// `BlockIODeviceWeight` property of systemd does not work properly, and systemd
// expects device path instead of major minor numbers, which is also confusing
// for users. So we use fs work around for now.
func joinBlkio(c *configs.Cgroup, pid int) error {
	path, err := getSubsystemPath(c, "blkio")
	if err != nil {
		return err
	}
	if c.BlkioWeightDevice != "" {
		if err := writeFile(path, "blkio.weight_device", c.BlkioWeightDevice); err != nil {
			return err
		}
	}
	if c.BlkioThrottleReadBpsDevice != "" {
		if err := writeFile(path, "blkio.throttle.read_bps_device", c.BlkioThrottleReadBpsDevice); err != nil {
			return err
		}
	}
	if c.BlkioThrottleWriteBpsDevice != "" {
		if err := writeFile(path, "blkio.throttle.write_bps_device", c.BlkioThrottleWriteBpsDevice); err != nil {
			return err
		}
	}
	if c.BlkioThrottleReadIOpsDevice != "" {
		if err := writeFile(path, "blkio.throttle.read_iops_device", c.BlkioThrottleReadIOpsDevice); err != nil {
			return err
		}
	}
	if c.BlkioThrottleWriteIOpsDevice != "" {
		if err := writeFile(path, "blkio.throttle.write_iops_device", c.BlkioThrottleWriteIOpsDevice); err != nil {
			return err
		}
	}

	return nil
}
