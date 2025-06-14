package utils

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const (
	// PCI address consists of:
	// - domain:  16 bits, it can be represented in 4-digit hexadecimal number
	// - bus:      8 bits, it can be represented in 2-digit hexadecimal number
	// - device:   5 bits, it can be represented in 2-digit hexadecimal number
	// - function: 3 bits, it can be represented in 1-digit hexadecimal number
	//
	// ref: "Chapter 12. PCI Drivers - PCI Addressing"
	//      Linux Device Drivers, 3rd Edition by Jonathan Corbet, Alessandro Rubini, Greg Kroah-Hartman
	//      https://www.oreilly.com/library/view/linux-device-drivers/0596005903/ch12.html
	PCIDomainBits   = uint16(16)
	PCIDomainMax    = uint16(math.MaxUint16)
	PCIBusBits      = uint16(8)
	PCIBusMax       = uint16((1 << PCIBusBits) - 1)
	PCIDeviceBits   = uint16(5)
	PCIDeviceMax    = uint16((1 << PCIDeviceBits) - 1)
	PCIFunctionBits = uint16(3)
	PCIFunctionMax  = uint16((1 << PCIFunctionBits) - 1)
)

var (
	// bdfRegexp matches PCI address in BDF notation.
	// The format is <domain>:<bus>:<device>.<function>
	// where:
	// - domain: 4-digit hexadecimal number representing the PCI domain (16 bits)
	// - bus: 2-digit hexadecimal number representing the PCI bus (8 bits)
	// - device: 2-digit hexadecimal number representing the PCI device (5 bits)
	// - function: 1-digit hexadecimal number representing the PCI function (3 bits)
	//
	// Example: "0000:0e:1f.0" represents domain 0, bus 14, device 31, function 0.
	//
	// ref: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
	bdfRegexp = regexp.MustCompile(`^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f]{1})$`)

	// pciRootRegexp matches PCI root address in the format "pci<domain>:<bus>".
	// The format is:
	// - domain: 4-digit hexadecimal number representing the PCI domain (32 bits)
	// - bus: 2-digit hexadecimal number representing the PCI bus (8 bits)
	pciRootRegexp = regexp.MustCompile(`^([0-9a-f]{4}):([0-9a-f]{2})$`)
)

// PCIRoot represents a PCI root address in the combination of Domain and Bus.
// This represents the top-level element in sysfs PCI device hierarchy.
// (i.e. /sys/devices/pci<domain>:<bus>)
//
// ref: https://docs.kernel.org/PCI/sysfs-pci.html
type PCIRoot struct {
	domain uint16
	bus    uint16
}

func NewPCIRoot(domain, bus uint16) (*PCIRoot, error) {
	// no validation for domain, as it uses full 16 bits

	if bus > PCIBusMax {
		return nil, fmt.Errorf("invalid PCI bus number: %02x, must be in range 0-%d (%dbits)", bus, PCIBusMax, PCIBusBits)
	}
	return &PCIRoot{
		domain: domain,
		bus:    bus,
	}, nil
}

func MustNewPCIRoot(domain, bus uint16) *PCIRoot {
	root, err := NewPCIRoot(domain, bus)
	if err != nil {
		panic(fmt.Sprintf("failed to create PCIRoot: %v", err))
	}
	return root
}

// String returns the string representation of the PCIRoot in the format "domain:bus".
func (p *PCIRoot) String() string {
	return fmt.Sprintf("%04x:%02x", p.domain, p.bus)
}

// The PCIAddress holds PCI address components in BDF notation.
// <domain>:<bus>:<device>.<function>
// where:
// - domain: 4-digit hexadecimal number representing the PCI domain (16 bits)
// - bus: 2-digit hexadecimal number representing the PCI bus (8 bits)
// - device: 2-digit hexadecimal number representing the PCI device (5 bits)
// - function: 1-digit hexadecimal number representing the PCI function (3 bits)
//
// Example: "0000:0e:1f.0" represents domain 0, bus 14, device 31, function 0.
//
// ref:
// - BDF Notation: https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation
// - PCI Addressing: https://docs.kernel.org/PCI/sysfs-pci.html
type PCIAddress struct {
	domain   uint16
	bus      uint16
	device   uint16
	function uint16
}

func NewPCIAddress(domain, bus, device, function uint16) (*PCIAddress, error) {
	// no validation for domain, as it uses full 16 bits
	if bus > PCIBusMax {
		return nil, fmt.Errorf("invalid PCI bus number: %02x, must be in range 0-%d (%dbits)", bus, PCIBusMax, PCIBusBits)
	}
	if device > PCIDeviceMax {
		return nil, fmt.Errorf("invalid PCI device number: %02x, must be in range 0-%d (%dbits)", device, PCIDeviceMax, PCIDeviceBits)
	}
	if function > PCIFunctionMax {
		return nil, fmt.Errorf("invalid PCI function number: %01x, must be in range 0-%d (%dbits)", function, PCIFunctionMax, PCIFunctionBits)
	}
	return &PCIAddress{
		domain:   domain,
		bus:      bus,
		device:   device,
		function: function,
	}, nil
}

func MustNewPCIAddress(domain, bus, device, function uint16) *PCIAddress {
	addr, err := NewPCIAddress(domain, bus, device, function)
	if err != nil {
		panic(fmt.Sprintf("failed to create PCIAddress: %v", err))
	}
	return addr
}

func (p *PCIAddress) String() string {
	return fmt.Sprintf("%04x:%02x:%02x.%01x", p.domain, p.bus, p.device, p.function)
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
	pciRoot, err := ParsePCIRoot(
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

// ParsePCIRoot parses a PCIRoot from a string in the format "domain:bus".
// The format is:
// - domain: 4-digit hexadecimal number representing the PCI domain
// - bus: 2-digit hexadecimal number representing the PCI bus
// Example: "0000:01" represents domain 0, bus 1.
func ParsePCIRoot(str string) (*PCIRoot, error) {
	match := pciRootRegexp.FindStringSubmatch(str)

	if len(match) == 0 {
		return nil, fmt.Errorf("invalid PCIRoot format: %s", str)
	}

	parsePart := func(name, part string) (uint16, error) {
		value, err := parseHexTouint16(part)
		if err != nil {
			return 0, fmt.Errorf("invalid value %s in PCIRoot %s: %w", name, str, err)
		}
		return uint16(value), nil
	}

	var domain, bus uint16
	var err error
	if domain, err = parsePart("domain", match[1]); err != nil {
		return nil, err
	}
	if bus, err = parsePart("bus", match[2]); err != nil {
		return nil, err
	}

	return NewPCIRoot(domain, bus)
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

	parsePart := func(name, part string) (uint16, error) {
		value, err := parseHexTouint16(part)
		if err != nil {
			return 0, fmt.Errorf("invalid value %s in PCI address %s: %w", name, bdfString, err)
		}
		return uint16(value), nil
	}

	var domain, bus, device, function uint16
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

	return NewPCIAddress(domain, bus, device, function)
}

func parseHexTouint16(s string) (uint16, error) {
	value, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return 0, err
	}
	return uint16(value), nil
}
