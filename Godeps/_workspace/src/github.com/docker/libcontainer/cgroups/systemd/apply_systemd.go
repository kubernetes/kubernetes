// +build linux

package systemd

import (
	"bytes"
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
	"github.com/godbus/dbus"
)

type systemdCgroup struct {
	cgroup *cgroups.Cgroup
}

type subsystem interface {
	GetStats(string, *cgroups.Stats) error
}

var (
	connLock              sync.Mutex
	theConn               *systemd.Conn
	hasStartTransientUnit bool
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
				}
			}
		}
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

func Apply(c *cgroups.Cgroup, pid int) (map[string]string, error) {
	var (
		unitName   = getUnitName(c)
		slice      = "system.slice"
		properties []systemd.Property
		res        = &systemdCgroup{}
	)

	res.cgroup = c

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

	if c.Memory != 0 {
		properties = append(properties,
			newProp("MemoryLimit", uint64(c.Memory)))
	}
	// TODO: MemoryReservation and MemorySwap not available in systemd

	if c.CpuShares != 0 {
		properties = append(properties,
			newProp("CPUShares", uint64(c.CpuShares)))
	}

	if _, err := theConn.StartTransientUnit(unitName, "replace", properties...); err != nil {
		return nil, err
	}

	if !c.AllowAllDevices {
		if err := joinDevices(c, pid); err != nil {
			return nil, err
		}
	}

	// -1 disables memorySwap
	if c.MemorySwap >= 0 && (c.Memory != 0 || c.MemorySwap > 0) {
		if err := joinMemory(c, pid); err != nil {
			return nil, err
		}

	}

	// we need to manually join the freezer and cpuset cgroup in systemd
	// because it does not currently support it via the dbus api.
	if err := joinFreezer(c, pid); err != nil {
		return nil, err
	}

	if err := joinCpuset(c, pid); err != nil {
		return nil, err
	}

	paths := make(map[string]string)
	for _, sysname := range []string{
		"devices",
		"memory",
		"cpu",
		"cpuset",
		"cpuacct",
		"blkio",
		"perf_event",
		"freezer",
	} {
		subsystemPath, err := getSubsystemPath(res.cgroup, sysname)
		if err != nil {
			// Don't fail if a cgroup hierarchy was not found, just skip this subsystem
			if cgroups.IsNotFound(err) {
				continue
			}
			return nil, err
		}
		paths[sysname] = subsystemPath
	}
	return paths, nil
}

func writeFile(dir, file, data string) error {
	return ioutil.WriteFile(filepath.Join(dir, file), []byte(data), 0700)
}

func joinFreezer(c *cgroups.Cgroup, pid int) error {
	path, err := getSubsystemPath(c, "freezer")
	if err != nil {
		return err
	}

	if err := os.MkdirAll(path, 0755); err != nil && !os.IsExist(err) {
		return err
	}

	return ioutil.WriteFile(filepath.Join(path, "cgroup.procs"), []byte(strconv.Itoa(pid)), 0700)
}

func getSubsystemPath(c *cgroups.Cgroup, subsystem string) (string, error) {
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

func Freeze(c *cgroups.Cgroup, state cgroups.FreezerState) error {
	path, err := getSubsystemPath(c, "freezer")
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile(filepath.Join(path, "freezer.state"), []byte(state), 0); err != nil {
		return err
	}
	for {
		state_, err := ioutil.ReadFile(filepath.Join(path, "freezer.state"))
		if err != nil {
			return err
		}
		if string(state) == string(bytes.TrimSpace(state_)) {
			break
		}
		time.Sleep(1 * time.Millisecond)
	}
	return nil
}

func GetPids(c *cgroups.Cgroup) ([]int, error) {
	path, err := getSubsystemPath(c, "cpu")
	if err != nil {
		return nil, err
	}

	return cgroups.ReadProcsFile(path)
}

func getUnitName(c *cgroups.Cgroup) string {
	return fmt.Sprintf("%s-%s.scope", c.Parent, c.Name)
}

// Atm we can't use the systemd device support because of two missing things:
// * Support for wildcards to allow mknod on any device
// * Support for wildcards to allow /dev/pts support
//
// The second is available in more recent systemd as "char-pts", but not in e.g. v208 which is
// in wide use. When both these are availalable we will be able to switch, but need to keep the old
// implementation for backwards compat.
//
// Note: we can't use systemd to set up the initial limits, and then change the cgroup
// because systemd will re-write the device settings if it needs to re-apply the cgroup context.
// This happens at least for v208 when any sibling unit is started.
func joinDevices(c *cgroups.Cgroup, pid int) error {
	path, err := getSubsystemPath(c, "devices")
	if err != nil {
		return err
	}

	if err := os.MkdirAll(path, 0755); err != nil && !os.IsExist(err) {
		return err
	}

	if err := ioutil.WriteFile(filepath.Join(path, "cgroup.procs"), []byte(strconv.Itoa(pid)), 0700); err != nil {
		return err
	}

	if err := writeFile(path, "devices.deny", "a"); err != nil {
		return err
	}

	for _, dev := range c.AllowedDevices {
		if err := writeFile(path, "devices.allow", dev.GetCgroupAllowString()); err != nil {
			return err
		}
	}

	return nil
}

// Symmetrical public function to update device based cgroups.  Also available
// in the fs implementation.
func ApplyDevices(c *cgroups.Cgroup, pid int) error {
	return joinDevices(c, pid)
}

func joinMemory(c *cgroups.Cgroup, pid int) error {
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
func joinCpuset(c *cgroups.Cgroup, pid int) error {
	path, err := getSubsystemPath(c, "cpuset")
	if err != nil {
		return err
	}

	s := &fs.CpusetGroup{}

	return s.SetDir(path, c.CpusetCpus, c.CpusetMems, pid)
}
