package systemd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	dbus "github.com/godbus/dbus/v5"
	"github.com/sirupsen/logrus"

	cgroupdevices "github.com/opencontainers/runc/libcontainer/cgroups/devices"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/devices"
)

const (
	// Default kernel value for cpu quota period is 100000 us (100 ms), same for v1 and v2.
	// v1: https://www.kernel.org/doc/html/latest/scheduler/sched-bwc.html and
	// v2: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
	defCPUQuotaPeriod = uint64(100000)
)

var (
	versionOnce sync.Once
	version     int

	isRunningSystemdOnce sync.Once
	isRunningSystemd     bool
)

// NOTE: This function comes from package github.com/coreos/go-systemd/util
// It was borrowed here to avoid a dependency on cgo.
//
// IsRunningSystemd checks whether the host was booted with systemd as its init
// system. This functions similarly to systemd's `sd_booted(3)`: internally, it
// checks whether /run/systemd/system/ exists and is a directory.
// http://www.freedesktop.org/software/systemd/man/sd_booted.html
func IsRunningSystemd() bool {
	isRunningSystemdOnce.Do(func() {
		fi, err := os.Lstat("/run/systemd/system")
		isRunningSystemd = err == nil && fi.IsDir()
	})
	return isRunningSystemd
}

// systemd represents slice hierarchy using `-`, so we need to follow suit when
// generating the path of slice. Essentially, test-a-b.slice becomes
// /test.slice/test-a.slice/test-a-b.slice.
func ExpandSlice(slice string) (string, error) {
	suffix := ".slice"
	// Name has to end with ".slice", but can't be just ".slice".
	if len(slice) < len(suffix) || !strings.HasSuffix(slice, suffix) {
		return "", fmt.Errorf("invalid slice name: %s", slice)
	}

	// Path-separators are not allowed.
	if strings.Contains(slice, "/") {
		return "", fmt.Errorf("invalid slice name: %s", slice)
	}

	var path, prefix string
	sliceName := strings.TrimSuffix(slice, suffix)
	// if input was -.slice, we should just return root now
	if sliceName == "-" {
		return "/", nil
	}
	for _, component := range strings.Split(sliceName, "-") {
		// test--a.slice isn't permitted, nor is -test.slice.
		if component == "" {
			return "", fmt.Errorf("invalid slice name: %s", slice)
		}

		// Append the component to the path and to the prefix.
		path += "/" + prefix + component + suffix
		prefix += component + "-"
	}
	return path, nil
}

func groupPrefix(ruleType devices.Type) (string, error) {
	switch ruleType {
	case devices.BlockDevice:
		return "block-", nil
	case devices.CharDevice:
		return "char-", nil
	default:
		return "", fmt.Errorf("device type %v has no group prefix", ruleType)
	}
}

// findDeviceGroup tries to find the device group name (as listed in
// /proc/devices) with the type prefixed as required for DeviceAllow, for a
// given (type, major) combination. If more than one device group exists, an
// arbitrary one is chosen.
func findDeviceGroup(ruleType devices.Type, ruleMajor int64) (string, error) {
	fh, err := os.Open("/proc/devices")
	if err != nil {
		return "", err
	}
	defer fh.Close()

	prefix, err := groupPrefix(ruleType)
	if err != nil {
		return "", err
	}

	scanner := bufio.NewScanner(fh)
	var currentType devices.Type
	for scanner.Scan() {
		// We need to strip spaces because the first number is column-aligned.
		line := strings.TrimSpace(scanner.Text())

		// Handle the "header" lines.
		switch line {
		case "Block devices:":
			currentType = devices.BlockDevice
			continue
		case "Character devices:":
			currentType = devices.CharDevice
			continue
		case "":
			continue
		}

		// Skip lines unrelated to our type.
		if currentType != ruleType {
			continue
		}

		// Parse out the (major, name).
		var (
			currMajor int64
			currName  string
		)
		if n, err := fmt.Sscanf(line, "%d %s", &currMajor, &currName); err != nil || n != 2 {
			if err == nil {
				err = errors.New("wrong number of fields")
			}
			return "", fmt.Errorf("scan /proc/devices line %q: %w", line, err)
		}

		if currMajor == ruleMajor {
			return prefix + currName, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("reading /proc/devices: %w", err)
	}
	// Couldn't find the device group.
	return "", nil
}

// DeviceAllow is the dbus type "a(ss)" which means we need a struct
// to represent it in Go.
type deviceAllowEntry struct {
	Path  string
	Perms string
}

func allowAllDevices() []systemdDbus.Property {
	// Setting mode to auto and removing all DeviceAllow rules
	// results in allowing access to all devices.
	return []systemdDbus.Property{
		newProp("DevicePolicy", "auto"),
		newProp("DeviceAllow", []deviceAllowEntry{}),
	}
}

// generateDeviceProperties takes the configured device rules and generates a
// corresponding set of systemd properties to configure the devices correctly.
func generateDeviceProperties(r *configs.Resources) ([]systemdDbus.Property, error) {
	if r.SkipDevices {
		return nil, nil
	}

	properties := []systemdDbus.Property{
		// Always run in the strictest white-list mode.
		newProp("DevicePolicy", "strict"),
		// Empty the DeviceAllow array before filling it.
		newProp("DeviceAllow", []deviceAllowEntry{}),
	}

	// Figure out the set of rules.
	configEmu := &cgroupdevices.Emulator{}
	for _, rule := range r.Devices {
		if err := configEmu.Apply(*rule); err != nil {
			return nil, fmt.Errorf("unable to apply rule for systemd: %w", err)
		}
	}
	// systemd doesn't support blacklists. So we log a warning, and tell
	// systemd to act as a deny-all whitelist. This ruleset will be replaced
	// with our normal fallback code. This may result in spurious errors, but
	// the only other option is to error out here.
	if configEmu.IsBlacklist() {
		// However, if we're dealing with an allow-all rule then we can do it.
		if configEmu.IsAllowAll() {
			return allowAllDevices(), nil
		}
		logrus.Warn("systemd doesn't support blacklist device rules -- applying temporary deny-all rule")
		return properties, nil
	}

	// Now generate the set of rules we actually need to apply. Unlike the
	// normal devices cgroup, in "strict" mode systemd defaults to a deny-all
	// whitelist which is the default for devices.Emulator.
	finalRules, err := configEmu.Rules()
	if err != nil {
		return nil, fmt.Errorf("unable to get simplified rules for systemd: %w", err)
	}
	var deviceAllowList []deviceAllowEntry
	for _, rule := range finalRules {
		if !rule.Allow {
			// Should never happen.
			return nil, fmt.Errorf("[internal error] cannot add deny rule to systemd DeviceAllow list: %v", *rule)
		}
		switch rule.Type {
		case devices.BlockDevice, devices.CharDevice:
		default:
			// Should never happen.
			return nil, fmt.Errorf("invalid device type for DeviceAllow: %v", rule.Type)
		}

		entry := deviceAllowEntry{
			Perms: string(rule.Permissions),
		}

		// systemd has a fairly odd (though understandable) syntax here, and
		// because of the OCI configuration format we have to do quite a bit of
		// trickery to convert things:
		//
		//  * Concrete rules with non-wildcard major/minor numbers have to use
		//    /dev/{block,char} paths. This is slightly odd because it means
		//    that we cannot add whitelist rules for devices that don't exist,
		//    but there's not too much we can do about that.
		//
		//    However, path globbing is not support for path-based rules so we
		//    need to handle wildcards in some other manner.
		//
		//  * Wildcard-minor rules have to specify a "device group name" (the
		//    second column in /proc/devices).
		//
		//  * Wildcard (major and minor) rules can just specify a glob with the
		//    type ("char-*" or "block-*").
		//
		// The only type of rule we can't handle is wildcard-major rules, and
		// so we'll give a warning in that case (note that the fallback code
		// will insert any rules systemd couldn't handle). What amazing fun.

		if rule.Major == devices.Wildcard {
			// "_ *:n _" rules aren't supported by systemd.
			if rule.Minor != devices.Wildcard {
				logrus.Warnf("systemd doesn't support '*:n' device rules -- temporarily ignoring rule: %v", *rule)
				continue
			}

			// "_ *:* _" rules just wildcard everything.
			prefix, err := groupPrefix(rule.Type)
			if err != nil {
				return nil, err
			}
			entry.Path = prefix + "*"
		} else if rule.Minor == devices.Wildcard {
			// "_ n:* _" rules require a device group from /proc/devices.
			group, err := findDeviceGroup(rule.Type, rule.Major)
			if err != nil {
				return nil, fmt.Errorf("unable to find device '%v/%d': %w", rule.Type, rule.Major, err)
			}
			if group == "" {
				// Couldn't find a group.
				logrus.Warnf("could not find device group for '%v/%d' in /proc/devices -- temporarily ignoring rule: %v", rule.Type, rule.Major, *rule)
				continue
			}
			entry.Path = group
		} else {
			// "_ n:m _" rules are just a path in /dev/{block,char}/.
			switch rule.Type {
			case devices.BlockDevice:
				entry.Path = fmt.Sprintf("/dev/block/%d:%d", rule.Major, rule.Minor)
			case devices.CharDevice:
				entry.Path = fmt.Sprintf("/dev/char/%d:%d", rule.Major, rule.Minor)
			}
			// systemd will issue a warning if the path we give here doesn't exist.
			// Since all of this logic is best-effort anyway (we manually set these
			// rules separately to systemd) we can safely skip entries that don't
			// have a corresponding path.
			if _, err := os.Stat(entry.Path); err != nil {
				// Also check /sys/dev so that we don't depend on /dev/{block,char}
				// being populated. (/dev/{block,char} is populated by udev, which
				// isn't strictly required for systemd). Ironically, this happens most
				// easily when starting containerd within a runc created container
				// itself.

				// We don't bother with securejoin here because we create entry.Path
				// right above here, so we know it's safe.
				if _, err := os.Stat("/sys" + entry.Path); err != nil {
					logrus.Warnf("skipping device %s for systemd: %s", entry.Path, err)
					continue
				}
			}
		}
		deviceAllowList = append(deviceAllowList, entry)
	}

	properties = append(properties, newProp("DeviceAllow", deviceAllowList))
	return properties, nil
}

func newProp(name string, units interface{}) systemdDbus.Property {
	return systemdDbus.Property{
		Name:  name,
		Value: dbus.MakeVariant(units),
	}
}

func getUnitName(c *configs.Cgroup) string {
	// by default, we create a scope unless the user explicitly asks for a slice.
	if !strings.HasSuffix(c.Name, ".slice") {
		return c.ScopePrefix + "-" + c.Name + ".scope"
	}
	return c.Name
}

// This code should be in sync with getUnitName.
func getUnitType(unitName string) string {
	if strings.HasSuffix(unitName, ".slice") {
		return "Slice"
	}
	return "Scope"
}

// isDbusError returns true if the error is a specific dbus error.
func isDbusError(err error, name string) bool {
	if err != nil {
		var derr dbus.Error
		if errors.As(err, &derr) {
			return strings.Contains(derr.Name, name)
		}
	}
	return false
}

// isUnitExists returns true if the error is that a systemd unit already exists.
func isUnitExists(err error) bool {
	return isDbusError(err, "org.freedesktop.systemd1.UnitExists")
}

func startUnit(cm *dbusConnManager, unitName string, properties []systemdDbus.Property) error {
	statusChan := make(chan string, 1)
	err := cm.retryOnDisconnect(func(c *systemdDbus.Conn) error {
		_, err := c.StartTransientUnitContext(context.TODO(), unitName, "replace", properties, statusChan)
		return err
	})
	if err == nil {
		timeout := time.NewTimer(30 * time.Second)
		defer timeout.Stop()

		select {
		case s := <-statusChan:
			close(statusChan)
			// Please refer to https://pkg.go.dev/github.com/coreos/go-systemd/v22/dbus#Conn.StartUnit
			if s != "done" {
				resetFailedUnit(cm, unitName)
				return fmt.Errorf("error creating systemd unit `%s`: got `%s`", unitName, s)
			}
		case <-timeout.C:
			resetFailedUnit(cm, unitName)
			return errors.New("Timeout waiting for systemd to create " + unitName)
		}
	} else if !isUnitExists(err) {
		return err
	}

	return nil
}

func stopUnit(cm *dbusConnManager, unitName string) error {
	statusChan := make(chan string, 1)
	err := cm.retryOnDisconnect(func(c *systemdDbus.Conn) error {
		_, err := c.StopUnitContext(context.TODO(), unitName, "replace", statusChan)
		return err
	})
	if err == nil {
		timeout := time.NewTimer(30 * time.Second)
		defer timeout.Stop()

		select {
		case s := <-statusChan:
			close(statusChan)
			// Please refer to https://godoc.org/github.com/coreos/go-systemd/v22/dbus#Conn.StartUnit
			if s != "done" {
				logrus.Warnf("error removing unit `%s`: got `%s`. Continuing...", unitName, s)
			}
		case <-timeout.C:
			return errors.New("Timed out while waiting for systemd to remove " + unitName)
		}
	}
	return nil
}

func resetFailedUnit(cm *dbusConnManager, name string) {
	err := cm.retryOnDisconnect(func(c *systemdDbus.Conn) error {
		return c.ResetFailedUnitContext(context.TODO(), name)
	})
	if err != nil {
		logrus.Warnf("unable to reset failed unit: %v", err)
	}
}

func getUnitTypeProperty(cm *dbusConnManager, unitName string, unitType string, propertyName string) (*systemdDbus.Property, error) {
	var prop *systemdDbus.Property
	err := cm.retryOnDisconnect(func(c *systemdDbus.Conn) (Err error) {
		prop, Err = c.GetUnitTypePropertyContext(context.TODO(), unitName, unitType, propertyName)
		return Err
	})
	return prop, err
}

func setUnitProperties(cm *dbusConnManager, name string, properties ...systemdDbus.Property) error {
	return cm.retryOnDisconnect(func(c *systemdDbus.Conn) error {
		return c.SetUnitPropertiesContext(context.TODO(), name, true, properties...)
	})
}

func getManagerProperty(cm *dbusConnManager, name string) (string, error) {
	str := ""
	err := cm.retryOnDisconnect(func(c *systemdDbus.Conn) error {
		var err error
		str, err = c.GetManagerProperty(name)
		return err
	})
	if err != nil {
		return "", err
	}
	return strconv.Unquote(str)
}

func systemdVersion(cm *dbusConnManager) int {
	versionOnce.Do(func() {
		version = -1
		verStr, err := getManagerProperty(cm, "Version")
		if err == nil {
			version, err = systemdVersionAtoi(verStr)
		}

		if err != nil {
			logrus.WithError(err).Error("unable to get systemd version")
		}
	})

	return version
}

func systemdVersionAtoi(verStr string) (int, error) {
	// verStr should be of the form:
	// "v245.4-1.fc32", "245", "v245-1.fc32", "245-1.fc32" (without quotes).
	// The result for all of the above should be 245.
	// Thus, we unconditionally remove the "v" prefix
	// and then match on the first integer we can grab.
	re := regexp.MustCompile(`v?([0-9]+)`)
	matches := re.FindStringSubmatch(verStr)
	if len(matches) < 2 {
		return 0, fmt.Errorf("can't parse version %s: incorrect number of matches %v", verStr, matches)
	}
	ver, err := strconv.Atoi(matches[1])
	if err != nil {
		return -1, fmt.Errorf("can't parse version: %w", err)
	}
	return ver, nil
}

func addCpuQuota(cm *dbusConnManager, properties *[]systemdDbus.Property, quota int64, period uint64) {
	if period != 0 {
		// systemd only supports CPUQuotaPeriodUSec since v242
		sdVer := systemdVersion(cm)
		if sdVer >= 242 {
			*properties = append(*properties,
				newProp("CPUQuotaPeriodUSec", period))
		} else {
			logrus.Debugf("systemd v%d is too old to support CPUQuotaPeriodSec "+
				" (setting will still be applied to cgroupfs)", sdVer)
		}
	}
	if quota != 0 || period != 0 {
		// corresponds to USEC_INFINITY in systemd
		cpuQuotaPerSecUSec := uint64(math.MaxUint64)
		if quota > 0 {
			if period == 0 {
				// assume the default
				period = defCPUQuotaPeriod
			}
			// systemd converts CPUQuotaPerSecUSec (microseconds per CPU second) to CPUQuota
			// (integer percentage of CPU) internally.  This means that if a fractional percent of
			// CPU is indicated by Resources.CpuQuota, we need to round up to the nearest
			// 10ms (1% of a second) such that child cgroups can set the cpu.cfs_quota_us they expect.
			cpuQuotaPerSecUSec = uint64(quota*1000000) / period
			if cpuQuotaPerSecUSec%10000 != 0 {
				cpuQuotaPerSecUSec = ((cpuQuotaPerSecUSec / 10000) + 1) * 10000
			}
		}
		*properties = append(*properties,
			newProp("CPUQuotaPerSecUSec", cpuQuotaPerSecUSec))
	}
}

func addCpuset(cm *dbusConnManager, props *[]systemdDbus.Property, cpus, mems string) error {
	if cpus == "" && mems == "" {
		return nil
	}

	// systemd only supports AllowedCPUs/AllowedMemoryNodes since v244
	sdVer := systemdVersion(cm)
	if sdVer < 244 {
		logrus.Debugf("systemd v%d is too old to support AllowedCPUs/AllowedMemoryNodes"+
			" (settings will still be applied to cgroupfs)", sdVer)
		return nil
	}

	if cpus != "" {
		bits, err := RangeToBits(cpus)
		if err != nil {
			return fmt.Errorf("resources.CPU.Cpus=%q conversion error: %w",
				cpus, err)
		}
		*props = append(*props,
			newProp("AllowedCPUs", bits))
	}
	if mems != "" {
		bits, err := RangeToBits(mems)
		if err != nil {
			return fmt.Errorf("resources.CPU.Mems=%q conversion error: %w",
				mems, err)
		}
		*props = append(*props,
			newProp("AllowedMemoryNodes", bits))
	}
	return nil
}
