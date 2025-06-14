package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var (
	// bdfRegexp matches PCI address in BDF notation.
	// The format is <domain>:<bus>:<device>.<function>
	// where:
	// - domain: 4-digit hexadecimal number representing the PCI domain
	// - bus: 2-digit hexadecimal number representing the PCI bus
	// - device: 2-digit hexadecimal number representing the PCI device
	// - function: 1-digit hexadecimal number representing the PCI function
	// Example: "0000:0e:1f.0" represents domain 0, bus 14, device 31, function 0.
	// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
	bdfRegexp = regexp.MustCompile(`^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f]{1})$`)
)

// PCIRoot represents a PCI root address in the combination of Domain and Bus.
// This represents the top-level element in sysfs PCI device hierarchy.
// (i.e. /sys/devices/pci<domain>:<bus>)
//
// ref: https://docs.kernel.org/PCI/sysfs-pci.html
type PCIRoot struct {
	Domain uint32
	Bus    uint32
}

func (p *PCIRoot) String() string {
	return fmt.Sprintf("%04x:%02x", p.Domain, p.Bus)
}

// The PCIAddress holds PCI address components in BDF notation.
// <domain>:<bus>:<device>.<function>
// where:
// - domain: 4-digit hexadecimal number representing the PCI domain
// - bus: 2-digit hexadecimal number representing the PCI bus
// - device: 2-digit hexadecimal number representing the PCI device
// - function: 1-digit hexadecimal number representing the PCI function
//
// Example: "0000:0e:1f.0" represents domain 0, bus 14, device 31, function 0.
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
type PCIAddress struct {
	Domain   uint32
	Bus      uint32
	Device   uint32
	Function uint32
}

func (p *PCIAddress) String() string {
	return fmt.Sprintf("%04x:%02x:%02x.%01x", p.Domain, p.Bus, p.Device, p.Function)
}

// ResolvePCIRoot resolves the PCI root for the PCIAddress.
// It returns a PCIRoot object that contains the domain and bus of the PCI root.
// The PCIRoot is derived from the sysfs path of the PCIAddress.
//
// sysfs argument is optional and only for testing purposes.
func (p *PCIAddress) ResolvePCIRoot(sysfs Sysfs) (*PCIRoot, error) {
	if sysfs == nil {
		sysfs = NewSysfs()
	}

	// e.g. /sys/devices/pci0000:01/...<intermediate PCI devices>.../0000:00:1f.0,
	sysDevicesPath, err := p.resolveSysDevicesPath(sysfs)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve sysfs path for PCI device %s: %w", p.String(), err)
	}

	pciRootPart := strings.Split(strings.TrimPrefix(sysDevicesPath, sysfs.Devices("")+"/"), "/")[0]
	pciRoot, err := parsePCIRoot(
		pciRootPart[3:], // skip "pci" prefix
	)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve PCI root for %s: %w", p.String(), err)
	}
	return pciRoot, nil
}

// resolveSysDevicesPath resolves the /sys/devices path for the PCIAddress.
//
// /sys/devices has directory structure which reflects the hardware hierarchy in the system.
// Therefore, the device path may contains intermediate directories(devices).
// Thus, we can not simply find the device path from the PCIAddress.
// Fortunately, /sys/bus/pci/devices/<address> is a symlink to the actual device path in /sys/devices.
// So we can resolve the actual device path by reading the symlink at /sys/bus/pci/devices/<address>.
//
// For example, if the PCIAddress is "0000:00:1f.0",
// /sys/bus/pci/devices/0000:00:1f.0 points to
// /sys/devices/pci0000:01/...<intermediate PCI devices>.../0000:00:1f.0,
func (p *PCIAddress) resolveSysDevicesPath(sysfs Sysfs) (string, error) {
	// e.g. /sys/bus/pci/devices/0000:00:1f.0
	sysBusPath := sysfs.Bus(filepath.Join("pci", "devices", p.String()))

	targetRelative, err := os.Readlink(sysBusPath)
	if err != nil {
		return "", fmt.Errorf("failed to read symlink for PCI device %s: %w", sysBusPath, err)
	}
	var targetAbs string
	if filepath.IsAbs(targetRelative) {
		targetAbs = targetRelative
	} else {
		// If the target is a relative path, we need to resolve it relative to the symlink's directory.
		targetAbs = filepath.Join(filepath.Dir(sysBusPath), targetRelative)
	}

	// targetAbs must be /sys/devices/pci0000:01/...<intermediate PCI devices>.../0000:00:1f.0
	devicePathPrefix := sysfs.Devices("pci")
	if !strings.HasPrefix(targetAbs, devicePathPrefix) || filepath.Base(targetAbs) != p.String() {
		return "", fmt.Errorf("invalid symlink target for PCI device %s: %s", sysBusPath, targetAbs)
	}

	return targetAbs, nil
}

// parsePCIRoot parses a PCIRoot from a string in the format "domain:bus".
// The format is:
// - domain: 4-digit hexadecimal number representing the PCI domain
// - bus: 2-digit hexadecimal number representing the PCI bus
// Example: "0000:01" represents domain 0, bus 1.
func parsePCIRoot(str string) (*PCIRoot, error) {
	parts := strings.Split(str, ":")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid PCIRoot format: %s", str)
	}

	parsePart := func(name, part string) (uint32, error) {
		value, err := parseHexToUint32(part)
		if err != nil {
			return 0, fmt.Errorf("invalid value %s in PCIRoot %s: %w", name, str, err)
		}
		return uint32(value), nil
	}

	var domain, bus uint32
	var err error
	if domain, err = parsePart("domain", parts[0]); err != nil {
		return nil, err
	}
	if bus, err = parsePart("bus", parts[1]); err != nil {
		return nil, err
	}

	return &PCIRoot{
		Domain: domain,
		Bus:    bus,
	}, nil
}

// ParsePCIAddress parses a PCI address in BDF notation.
// The format is <domain>:<bus>:<device>.<function>
// where:
// - domain: 4-digit hexadecimal number representing the PCI domain
// - bus: 2-digit hexadecimal number representing the PCI bus
// - device: 2-digit hexadecimal number representing the PCI device
// - function: 1-digit hexadecimal number representing the PCI function
// The function returns a PCIAddress object or an error if the format is invalid.
//
// Example: "0000:0e:1f.0" represents domain 0, bus 14, device 31, function 0.
//
// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
func ParsePCIAddress(bdfString string) (*PCIAddress, error) {
	match := bdfRegexp.FindStringSubmatch(bdfString)

	if len(match) == 0 {
		return nil, fmt.Errorf("invalid PCI address format: %s", bdfString)
	}

	parsePart := func(name, part string) (uint32, error) {
		value, err := parseHexToUint32(part)
		if err != nil {
			return 0, fmt.Errorf("invalid value %s in PCI address %s: %w", name, bdfString, err)
		}
		return uint32(value), nil
	}

	var domain, bus, device, function uint32
	var err error
	if domain, err = parsePart("domain", match[1]); err != nil {
		return nil, err
	}
	if bus, err = parsePart("bus", match[2]); err != nil {
		return nil, err
	}
	if device, err = parsePart("device", match[3]); err != nil {
		return nil, err
	}
	if function, err = parsePart("function", match[4]); err != nil {
		return nil, err
	}

	return &PCIAddress{
		Domain:   uint32(domain),
		Bus:      uint32(bus),
		Device:   uint32(device),
		Function: uint32(function),
	}, nil
}

func parseHexToUint32(s string) (uint32, error) {
	value, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return 0, err
	}
	return uint32(value), nil
}
