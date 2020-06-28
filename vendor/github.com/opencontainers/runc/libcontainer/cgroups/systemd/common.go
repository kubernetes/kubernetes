package systemd

import (
	"bufio"
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
	"github.com/opencontainers/runc/libcontainer/cgroups/devices"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

var (
	connOnce sync.Once
	connDbus *systemdDbus.Conn
	connErr  error

	versionOnce sync.Once
	version     int
	versionErr  error
)

// NOTE: This function comes from package github.com/coreos/go-systemd/util
// It was borrowed here to avoid a dependency on cgo.
//
// IsRunningSystemd checks whether the host was booted with systemd as its init
// system. This functions similarly to systemd's `sd_booted(3)`: internally, it
// checks whether /run/systemd/system/ exists and is a directory.
// http://www.freedesktop.org/software/systemd/man/sd_booted.html
func IsRunningSystemd() bool {
	fi, err := os.Lstat("/run/systemd/system")
	if err != nil {
		return false
	}
	return fi.IsDir()
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

func groupPrefix(ruleType configs.DeviceType) (string, error) {
	switch ruleType {
	case configs.BlockDevice:
		return "block-", nil
	case configs.CharDevice:
		return "char-", nil
	default:
		return "", errors.Errorf("device type %v has no group prefix", ruleType)
	}
}

// findDeviceGroup tries to find the device group name (as listed in
// /proc/devices) with the type prefixed as requried for DeviceAllow, for a
// given (type, major) combination. If more than one device group exists, an
// arbitrary one is chosen.
func findDeviceGroup(ruleType configs.DeviceType, ruleMajor int64) (string, error) {
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
	var currentType configs.DeviceType
	for scanner.Scan() {
		// We need to strip spaces because the first number is column-aligned.
		line := strings.TrimSpace(scanner.Text())

		// Handle the "header" lines.
		switch line {
		case "Block devices:":
			currentType = configs.BlockDevice
			continue
		case "Character devices:":
			currentType = configs.CharDevice
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
				err = errors.Errorf("wrong number of fields")
			}
			return "", errors.Wrapf(err, "scan /proc/devices line %q", line)
		}

		if currMajor == ruleMajor {
			return prefix + currName, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", errors.Wrap(err, "reading /proc/devices")
	}
	// Couldn't find the device group.
	return "", nil
}

// generateDeviceProperties takes the configured device rules and generates a
// corresponding set of systemd properties to configure the devices correctly.
func generateDeviceProperties(rules []*configs.DeviceRule) ([]systemdDbus.Property, error) {
	// DeviceAllow is the type "a(ss)" which means we need a temporary struct
	// to represent it in Go.
	type deviceAllowEntry struct {
		Path  string
		Perms string
	}

	properties := []systemdDbus.Property{
		// Always run in the strictest white-list mode.
		newProp("DevicePolicy", "strict"),
		// Empty the DeviceAllow array before filling it.
		newProp("DeviceAllow", []deviceAllowEntry{}),
	}

	// Figure out the set of rules.
	configEmu := &devices.Emulator{}
	for _, rule := range rules {
		if err := configEmu.Apply(*rule); err != nil {
			return nil, errors.Wrap(err, "apply rule for systemd")
		}
	}
	// systemd doesn't support blacklists. So we log a warning, and tell
	// systemd to act as a deny-all whitelist. This ruleset will be replaced
	// with our normal fallback code. This may result in spurrious errors, but
	// the only other option is to error out here.
	if configEmu.IsBlacklist() {
		// However, if we're dealing with an allow-all rule then we can do it.
		if configEmu.IsAllowAll() {
			return []systemdDbus.Property{
				// Run in white-list mode by setting to "auto" and removing all
				// DeviceAllow rules.
				newProp("DevicePolicy", "auto"),
				newProp("DeviceAllow", []deviceAllowEntry{}),
			}, nil
		}
		logrus.Warn("systemd doesn't support blacklist device rules -- applying temporary deny-all rule")
		return properties, nil
	}

	// Now generate the set of rules we actually need to apply. Unlike the
	// normal devices cgroup, in "strict" mode systemd defaults to a deny-all
	// whitelist which is the default for devices.Emulator.
	baseEmu := &devices.Emulator{}
	finalRules, err := baseEmu.Transition(configEmu)
	if err != nil {
		return nil, errors.Wrap(err, "get simplified rules for systemd")
	}
	var deviceAllowList []deviceAllowEntry
	for _, rule := range finalRules {
		if !rule.Allow {
			// Should never happen.
			return nil, errors.Errorf("[internal error] cannot add deny rule to systemd DeviceAllow list: %v", *rule)
		}
		switch rule.Type {
		case configs.BlockDevice, configs.CharDevice:
		default:
			// Should never happen.
			return nil, errors.Errorf("invalid device type for DeviceAllow: %v", rule.Type)
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

		if rule.Major == configs.Wildcard {
			// "_ *:n _" rules aren't supported by systemd.
			if rule.Minor != configs.Wildcard {
				logrus.Warnf("systemd doesn't support '*:n' device rules -- temporarily ignoring rule: %v", *rule)
				continue
			}

			// "_ *:* _" rules just wildcard everything.
			prefix, err := groupPrefix(rule.Type)
			if err != nil {
				return nil, err
			}
			entry.Path = prefix + "*"
		} else if rule.Minor == configs.Wildcard {
			// "_ n:* _" rules require a device group from /proc/devices.
			group, err := findDeviceGroup(rule.Type, rule.Major)
			if err != nil {
				return nil, errors.Wrapf(err, "find device '%v/%d'", rule.Type, rule.Major)
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
			case configs.BlockDevice:
				entry.Path = fmt.Sprintf("/dev/block/%d:%d", rule.Major, rule.Minor)
			case configs.CharDevice:
				entry.Path = fmt.Sprintf("/dev/char/%d:%d", rule.Major, rule.Minor)
			}
		}
		deviceAllowList = append(deviceAllowList, entry)
	}

	properties = append(properties, newProp("DeviceAllow", deviceAllowList))
	return properties, nil
}

// getDbusConnection lazy initializes systemd dbus connection
// and returns it
func getDbusConnection(rootless bool) (*systemdDbus.Conn, error) {
	connOnce.Do(func() {
		if rootless {
			connDbus, connErr = NewUserSystemdDbus()
		} else {
			connDbus, connErr = systemdDbus.New()
		}
	})
	return connDbus, connErr
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
		return fmt.Sprintf("%s-%s.scope", c.ScopePrefix, c.Name)
	}
	return c.Name
}

// isUnitExists returns true if the error is that a systemd unit already exists.
func isUnitExists(err error) bool {
	if err != nil {
		if dbusError, ok := err.(dbus.Error); ok {
			return strings.Contains(dbusError.Name, "org.freedesktop.systemd1.UnitExists")
		}
	}
	return false
}

func startUnit(dbusConnection *systemdDbus.Conn, unitName string, properties []systemdDbus.Property) error {
	statusChan := make(chan string, 1)
	if _, err := dbusConnection.StartTransientUnit(unitName, "replace", properties, statusChan); err == nil {
		select {
		case s := <-statusChan:
			close(statusChan)
			// Please refer to https://godoc.org/github.com/coreos/go-systemd/dbus#Conn.StartUnit
			if s != "done" {
				dbusConnection.ResetFailedUnit(unitName)
				return errors.Errorf("error creating systemd unit `%s`: got `%s`", unitName, s)
			}
		case <-time.After(time.Second):
			logrus.Warnf("Timed out while waiting for StartTransientUnit(%s) completion signal from dbus. Continuing...", unitName)
		}
	} else if !isUnitExists(err) {
		return err
	}

	return nil
}

func stopUnit(dbusConnection *systemdDbus.Conn, unitName string) error {
	statusChan := make(chan string, 1)
	if _, err := dbusConnection.StopUnit(unitName, "replace", statusChan); err == nil {
		select {
		case s := <-statusChan:
			close(statusChan)
			// Please refer to https://godoc.org/github.com/coreos/go-systemd/dbus#Conn.StartUnit
			if s != "done" {
				logrus.Warnf("error removing unit `%s`: got `%s`. Continuing...", unitName, s)
			}
		case <-time.After(time.Second):
			logrus.Warnf("Timed out while waiting for StopUnit(%s) completion signal from dbus. Continuing...", unitName)
		}
	}
	return nil
}

func systemdVersion(conn *systemdDbus.Conn) (int, error) {
	versionOnce.Do(func() {
		version = -1
		verStr, err := conn.GetManagerProperty("Version")
		if err != nil {
			versionErr = err
			return
		}

		version, versionErr = systemdVersionAtoi(verStr)
		return
	})

	return version, versionErr
}

func systemdVersionAtoi(verStr string) (int, error) {
	// verStr should be of the form:
	// "v245.4-1.fc32", "245", "v245-1.fc32", "245-1.fc32"
	// all the input strings include quotes, and the output int should be 245
	// thus, we unconditionally remove the `"v`
	// and then match on the first integer we can grab
	re := regexp.MustCompile(`"?v?([0-9]+)`)
	matches := re.FindStringSubmatch(verStr)
	if len(matches) < 2 {
		return 0, errors.Errorf("can't parse version %s: incorrect number of matches %v", verStr, matches)
	}
	ver, err := strconv.Atoi(matches[1])
	return ver, errors.Wrapf(err, "can't parse version %s", verStr)
}

func addCpuQuota(conn *systemdDbus.Conn, properties *[]systemdDbus.Property, quota int64, period uint64) {
	if period != 0 {
		// systemd only supports CPUQuotaPeriodUSec since v242
		sdVer, err := systemdVersion(conn)
		if err != nil {
			logrus.Warnf("systemdVersion: %s", err)
		} else if sdVer >= 242 {
			*properties = append(*properties,
				newProp("CPUQuotaPeriodUSec", period))
		}
	}
	if quota != 0 || period != 0 {
		// corresponds to USEC_INFINITY in systemd
		cpuQuotaPerSecUSec := uint64(math.MaxUint64)
		if quota > 0 {
			if period == 0 {
				// assume the default kernel value of 100000 us (100 ms), same for v1 and v2.
				// v1: https://www.kernel.org/doc/html/latest/scheduler/sched-bwc.html and
				// v2: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
				period = 100000
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
