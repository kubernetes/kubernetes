package systemd

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/sirupsen/logrus"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fs2"
)

const (
	cpuIdleSupportedVersion = 252
)

type UnifiedManager struct {
	mu      sync.Mutex
	cgroups *cgroups.Cgroup
	// path is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope"
	path  string
	dbus  *dbusConnManager
	fsMgr cgroups.Manager
}

func NewUnifiedManager(config *cgroups.Cgroup, path string) (*UnifiedManager, error) {
	m := &UnifiedManager{
		cgroups: config,
		path:    path,
		dbus:    newDbusConnManager(config.Rootless),
	}
	if err := m.initPath(); err != nil {
		return nil, err
	}

	fsMgr, err := fs2.NewManager(config, m.path)
	if err != nil {
		return nil, err
	}
	m.fsMgr = fsMgr

	return m, nil
}

func shouldSetCPUIdle(cm *dbusConnManager, v string) bool {
	// The only valid values for cpu.idle are 0 and 1. As it is
	// not possible to directly set cpu.idle to 0 via systemd,
	// ignore 0. Ignore other values as we'll error out later
	// in Set() while calling fsMgr.Set().
	return v == "1" && systemdVersion(cm) >= cpuIdleSupportedVersion
}

// unifiedResToSystemdProps tries to convert from Cgroup.Resources.Unified
// key/value map (where key is cgroupfs file name) to systemd unit properties.
// This is on a best-effort basis, so the properties that are not known
// (to this function and/or systemd) are ignored (but logged with "debug"
// log level).
//
// For the list of keys, see https://www.kernel.org/doc/Documentation/cgroup-v2.txt
//
// For the list of systemd unit properties, see systemd.resource-control(5).
func unifiedResToSystemdProps(cm *dbusConnManager, res map[string]string) (props []systemdDbus.Property, _ error) {
	var err error

	for k, v := range res {
		if strings.Contains(k, "/") {
			return nil, fmt.Errorf("unified resource %q must be a file name (no slashes)", k)
		}
		if strings.IndexByte(k, '.') <= 0 {
			return nil, fmt.Errorf("unified resource %q must be in the form CONTROLLER.PARAMETER", k)
		}
		// Kernel is quite forgiving to extra whitespace
		// around the value, and so should we.
		v = strings.TrimSpace(v)
		// Please keep cases in alphabetical order.
		switch k {
		case "cpu.idle":
			if shouldSetCPUIdle(cm, v) {
				// Setting CPUWeight to 0 tells systemd
				// to set cpu.idle to 1.
				props = append(props,
					newProp("CPUWeight", uint64(0)))
			}

		case "cpu.max":
			// value: quota [period]
			quota := int64(0) // 0 means "unlimited" for addCpuQuota, if period is set
			period := defCPUQuotaPeriod
			sv := strings.Fields(v)
			if len(sv) < 1 || len(sv) > 2 {
				return nil, fmt.Errorf("unified resource %q value invalid: %q", k, v)
			}
			// quota
			if sv[0] != "max" {
				quota, err = strconv.ParseInt(sv[0], 10, 64)
				if err != nil {
					return nil, fmt.Errorf("unified resource %q period value conversion error: %w", k, err)
				}
			}
			// period
			if len(sv) == 2 {
				period, err = strconv.ParseUint(sv[1], 10, 64)
				if err != nil {
					return nil, fmt.Errorf("unified resource %q quota value conversion error: %w", k, err)
				}
			}
			addCPUQuota(cm, &props, &quota, period)

		case "cpu.weight":
			if shouldSetCPUIdle(cm, strings.TrimSpace(res["cpu.idle"])) {
				// Do not add duplicate CPUWeight property
				// (see case "cpu.idle" above).
				logrus.Warn("unable to apply both cpu.weight and cpu.idle to systemd, ignoring cpu.weight")
				continue
			}
			num, err := strconv.ParseUint(v, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("unified resource %q value conversion error: %w", k, err)
			}
			props = append(props,
				newProp("CPUWeight", num))

		case "cpuset.cpus", "cpuset.mems":
			bits, err := RangeToBits(v)
			if err != nil {
				return nil, fmt.Errorf("unified resource %q=%q conversion error: %w", k, v, err)
			}
			m := map[string]string{
				"cpuset.cpus": "AllowedCPUs",
				"cpuset.mems": "AllowedMemoryNodes",
			}
			// systemd only supports these properties since v244
			sdVer := systemdVersion(cm)
			if sdVer >= 244 {
				props = append(props,
					newProp(m[k], bits))
			} else {
				logrus.Debugf("systemd v%d is too old to support %s"+
					" (setting will still be applied to cgroupfs)",
					sdVer, m[k])
			}

		case "memory.high", "memory.low", "memory.min", "memory.max", "memory.swap.max":
			num := uint64(math.MaxUint64)
			if v != "max" {
				num, err = strconv.ParseUint(v, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("unified resource %q value conversion error: %w", k, err)
				}
			}
			m := map[string]string{
				"memory.high":     "MemoryHigh",
				"memory.low":      "MemoryLow",
				"memory.min":      "MemoryMin",
				"memory.max":      "MemoryMax",
				"memory.swap.max": "MemorySwapMax",
			}
			props = append(props,
				newProp(m[k], num))

		case "pids.max":
			num := uint64(math.MaxUint64)
			if v != "max" {
				var err error
				num, err = strconv.ParseUint(v, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("unified resource %q value conversion error: %w", k, err)
				}
			}
			props = append(props,
				newProp("TasksMax", num))

		case "memory.oom.group":
			// Setting this to 1 is roughly equivalent to OOMPolicy=kill
			// (as per systemd.service(5) and
			// https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html),
			// but it's not clear what to do if it is unset or set
			// to 0 in runc update, as there are two other possible
			// values for OOMPolicy (continue/stop).
			fallthrough

		default:
			// Ignore the unknown resource here -- will still be
			// applied in Set which calls fs2.Set.
			logrus.Debugf("don't know how to convert unified resource %q=%q to systemd unit property; skipping (will still be applied to cgroupfs)", k, v)
		}
	}

	return props, nil
}

func genV2ResourcesProperties(dirPath string, r *cgroups.Resources, cm *dbusConnManager) ([]systemdDbus.Property, error) {
	// We need this check before setting systemd properties, otherwise
	// the container is OOM-killed and the systemd unit is removed
	// before we get to fsMgr.Set().
	if err := fs2.CheckMemoryUsage(dirPath, r); err != nil {
		return nil, err
	}

	var properties []systemdDbus.Property

	// NOTE: This is of questionable correctness because we insert our own
	//       devices eBPF program later. Two programs with identical rules
	//       aren't the end of the world, but it is a bit concerning. However
	//       it's unclear if systemd removes all eBPF programs attached when
	//       doing SetUnitProperties...
	deviceProperties, err := generateDeviceProperties(r, cm)
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

	idleSet := false
	// The logic here is the same as in shouldSetCPUIdle.
	if r.CPUIdle != nil && *r.CPUIdle == 1 && systemdVersion(cm) >= cpuIdleSupportedVersion {
		properties = append(properties,
			newProp("CPUWeight", uint64(0)))
		idleSet = true
	}
	if r.CpuWeight != 0 {
		if idleSet {
			// Ignore CpuWeight if CPUIdle is already set.
			logrus.Warn("unable to apply both CPUWeight and CpuIdle to systemd, ignoring CPUWeight")
		} else {
			properties = append(properties,
				newProp("CPUWeight", r.CpuWeight))
		}
	}

	addCPUQuota(cm, &properties, &r.CpuQuota, r.CpuPeriod)

	if r.PidsLimit > 0 || r.PidsLimit == -1 {
		properties = append(properties,
			newProp("TasksMax", uint64(r.PidsLimit)))
	}

	err = addCpuset(cm, &properties, r.CpusetCpus, r.CpusetMems)
	if err != nil {
		return nil, err
	}

	// ignore r.KernelMemory

	// convert Resources.Unified map to systemd properties
	if r.Unified != nil {
		unifiedProps, err := unifiedResToSystemdProps(cm, r.Unified)
		if err != nil {
			return nil, err
		}
		properties = append(properties, unifiedProps...)
	}

	return properties, nil
}

func (m *UnifiedManager) Apply(pid int) error {
	var (
		c          = m.cgroups
		unitName   = getUnitName(c)
		properties []systemdDbus.Property
	)

	slice := "system.slice"
	if m.cgroups.Rootless {
		slice = "user.slice"
	}
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
		newProp("IOAccounting", true),
		newProp("TasksAccounting", true),
	)

	// Assume DefaultDependencies= will always work (the check for it was previously broken.)
	properties = append(properties,
		newProp("DefaultDependencies", false))

	properties = append(properties, c.SystemdProps...)

	if err := startUnit(m.dbus, unitName, properties, pid == -1); err != nil {
		return fmt.Errorf("unable to start unit %q (properties %+v): %w", unitName, properties, err)
	}

	if err := fs2.CreateCgroupPath(m.path, m.cgroups); err != nil {
		return err
	}

	if c.OwnerUID != nil {
		// The directory itself must be chowned.
		err := os.Chown(m.path, *c.OwnerUID, -1)
		if err != nil {
			return err
		}

		filesToChown, err := cgroupFilesToChown()
		if err != nil {
			return err
		}

		for _, v := range filesToChown {
			err := os.Chown(m.path+"/"+v, *c.OwnerUID, -1)
			// Some files might not be present.
			if err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}

	return nil
}

// The kernel exposes a list of files that should be chowned to the delegate
// uid in /sys/kernel/cgroup/delegate.  If the file is not present
// (Linux < 4.15), use the initial values mentioned in cgroups(7).
func cgroupFilesToChown() ([]string, error) {
	const cgroupDelegateFile = "/sys/kernel/cgroup/delegate"

	f, err := os.Open(cgroupDelegateFile)
	if err != nil {
		return []string{"cgroup.procs", "cgroup.subtree_control", "cgroup.threads"}, nil
	}
	defer f.Close()

	filesToChown := []string{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		filesToChown = append(filesToChown, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading %s: %w", cgroupDelegateFile, err)
	}

	return filesToChown, nil
}

func (m *UnifiedManager) Destroy() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	unitName := getUnitName(m.cgroups)
	if err := stopUnit(m.dbus, unitName); err != nil {
		return err
	}

	// systemd 239 do not remove sub-cgroups.
	err := m.fsMgr.Destroy()
	// fsMgr.Destroy has handled ErrNotExist
	if err != nil {
		return err
	}

	return nil
}

func (m *UnifiedManager) Path(_ string) string {
	return m.path
}

// getSliceFull value is used in initPath.
// The value is incompatible with systemdDbus.PropSlice.
func (m *UnifiedManager) getSliceFull() (string, error) {
	c := m.cgroups
	slice := "system.slice"
	if c.Rootless {
		slice = "user.slice"
	}
	if c.Parent != "" {
		var err error
		slice, err = ExpandSlice(c.Parent)
		if err != nil {
			return "", err
		}
	}

	if c.Rootless {
		// managerCG is typically "/user.slice/user-${uid}.slice/user@${uid}.service".
		managerCG, err := getManagerProperty(m.dbus, "ControlGroup")
		if err != nil {
			return "", err
		}
		slice = filepath.Join(managerCG, slice)
	}

	// an example of the final slice in rootless: "/user.slice/user-1001.slice/user@1001.service/user.slice"
	// NOTE: systemdDbus.PropSlice requires the "/user.slice/user-1001.slice/user@1001.service/" prefix NOT to be specified.
	return slice, nil
}

func (m *UnifiedManager) initPath() error {
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

func (m *UnifiedManager) Freeze(state cgroups.FreezerState) error {
	return m.fsMgr.Freeze(state)
}

func (m *UnifiedManager) GetPids() ([]int, error) {
	return cgroups.GetPids(m.path)
}

func (m *UnifiedManager) GetAllPids() ([]int, error) {
	return cgroups.GetAllPids(m.path)
}

func (m *UnifiedManager) GetStats() (*cgroups.Stats, error) {
	return m.fsMgr.GetStats()
}

func (m *UnifiedManager) Set(r *cgroups.Resources) error {
	if r == nil {
		return nil
	}
	// Use a copy since CpuQuota in r may be modified.
	rCopy := *r
	r = &rCopy
	properties, err := genV2ResourcesProperties(m.fsMgr.Path(""), r, m.dbus)
	if err != nil {
		return err
	}

	if err := setUnitProperties(m.dbus, getUnitName(m.cgroups), properties...); err != nil {
		return fmt.Errorf("unable to set unit properties: %w", err)
	}

	return m.fsMgr.Set(r)
}

func (m *UnifiedManager) GetPaths() map[string]string {
	paths := make(map[string]string, 1)
	paths[""] = m.path
	return paths
}

func (m *UnifiedManager) GetCgroups() (*cgroups.Cgroup, error) {
	return m.cgroups, nil
}

func (m *UnifiedManager) GetFreezerState() (cgroups.FreezerState, error) {
	return m.fsMgr.GetFreezerState()
}

func (m *UnifiedManager) Exists() bool {
	return cgroups.PathExists(m.path)
}

func (m *UnifiedManager) OOMKillCount() (uint64, error) {
	return m.fsMgr.OOMKillCount()
}
