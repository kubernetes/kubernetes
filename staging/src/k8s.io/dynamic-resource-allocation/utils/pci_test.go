package utils

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	dratesting "k8s.io/dynamic-resource-allocation/testing"
)

func TestPCIAddressString(t *testing.T) {
	tests := map[string]struct {
		address  PCIAddress
		expected string
	}{
		"zero":    {PCIAddress{Domain: 0, Bus: 0, Device: 0, Function: 0}, "0000:00:00.0"},
		"simple":  {PCIAddress{Domain: 1, Bus: 2, Device: 3, Function: 4}, "0001:02:03.4"},
		"complex": {PCIAddress{Domain: 0x1234, Bus: 0x56, Device: 0x78, Function: 0x9}, "1234:56:78.9"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if got := test.address.String(); got != test.expected {
				t.Errorf("Expected %s, got %s", test.expected, got)
			}
		})
	}
}

func TestParsePCIAddress(t *testing.T) {
	tests := map[string]struct {
		input      string
		expected   *PCIAddress
		expectsErr bool
	}{
		"zero":               {"0000:00:00.0", &PCIAddress{Domain: 0, Bus: 0, Device: 0, Function: 0}, false},
		"simple":             {"0001:02:03.4", &PCIAddress{Domain: 1, Bus: 2, Device: 3, Function: 4}, false},
		"complex":            {"1234:56:78.9", &PCIAddress{Domain: 0x1234, Bus: 0x56, Device: 0x78, Function: 0x9}, false},
		"invalid format":     {"0000:00:00", nil, true},
		"too many parts":     {"0000:00:00.0.1", nil, true},
		"invalid hex value":  {"0000:00:00.0a", nil, true},
		"completely invalid": {"invalid", nil, true},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			result, err := ParsePCIAddress(test.input)
			if (err != nil) != test.expectsErr {
				t.Errorf("ParsePCIAddress(%s) error = %v, wantErr %v", test.input, err, test.expectsErr)
				return
			}
			if !test.expectsErr && result.String() != test.expected.String() {
				t.Errorf("ParsePCIAddress(%s) = %s, want %s", test.input, result.String(), test.expected.String())
				return
			}
		})
	}
}

func TestPCIRootString(t *testing.T) {
	tests := map[string]struct {
		root     PCIRoot
		expected string
	}{
		"zero":    {PCIRoot{Domain: 0, Bus: 0}, "0000:00"},
		"simple":  {PCIRoot{Domain: 1, Bus: 2}, "0001:02"},
		"complex": {PCIRoot{Domain: 0x1234, Bus: 0x56}, "1234:56"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if got := test.root.String(); got != test.expected {
				t.Errorf("Expected %s, got %s", test.expected, got)
			}
		})
	}
}

func TestPCIAddressResolvePCIRoot(t *testing.T) {
	address := PCIAddress{
		Domain:   0x1234,
		Bus:      0x56,
		Device:   0x78,
		Function: 0x9,
	}
	root := PCIRoot{
		Domain: 0x2345,
		Bus:    0x67,
	}

	tests := map[string]struct {
		sysfs          func(*testing.T, string) Sysfs
		expectsErr     bool
		expectedErrMsg string
	}{
		"valid address": {
			sysfs: func(t *testing.T, testSysfsRoot string) Sysfs {
				sysfs := NewSysfsWithRoot(testSysfsRoot)
				devicePath := sysfs.Devices(filepath.Join("pci"+root.String(), address.String()))
				dratesting.TouchFile(t, devicePath)

				busPath := sysfs.Bus(filepath.Join("pci", "devices", address.String()))
				dratesting.CreateSymlink(t, devicePath, busPath)
				return sysfs
			},
		},
		"non-existent address": {
			sysfs: func(t *testing.T, testSysfsRoot string) Sysfs {
				return NewSysfsWithRoot(testSysfsRoot)
			},
			expectsErr:     true,
			expectedErrMsg: "no such file or directory",
		},
		"devices path not a symlink": {
			sysfs: func(t *testing.T, testSysfsRoot string) Sysfs {
				sysfs := NewSysfsWithRoot(testSysfsRoot)
				devicePath := sysfs.Devices(filepath.Join("pci"+root.String(), address.String()))
				dratesting.TouchFile(t, devicePath)

				// Create a file instead of a symlink
				busPath := sysfs.Bus(filepath.Join("pci", "devices", address.String()))
				dratesting.TouchFile(t, busPath)
				return sysfs
			},
			expectsErr:     true,
			expectedErrMsg: "invalid argument",
		},
		"invalid symlink target": {
			sysfs: func(t *testing.T, testSysfsRoot string) Sysfs {
				sysfs := NewSysfsWithRoot(testSysfsRoot)
				invalidDevicePath := sysfs.Devices(filepath.Join("invalid", "pci"+root.String(), address.String()))
				dratesting.TouchFile(t, invalidDevicePath)

				// Create a symlink that points to an invalid path
				busPath := sysfs.Bus(filepath.Join("pci", "devices", address.String()))
				dratesting.CreateSymlink(t, invalidDevicePath, busPath)
				return sysfs
			},
			expectsErr:     true,
			expectedErrMsg: "invalid symlink target for PCI device",
		},
		"invalid root format": {
			sysfs: func(t *testing.T, testSysfsRoot string) Sysfs {
				sysfs := NewSysfsWithRoot(testSysfsRoot)
				devicePath := sysfs.Devices(filepath.Join("pci-invalid-root", address.String()))
				dratesting.TouchFile(t, devicePath)

				// Create a symlink with an invalid root format
				busPath := sysfs.Bus(filepath.Join("pci", "devices", address.String()))
				dratesting.CreateSymlink(t, devicePath, busPath)
				return sysfs
			},
			expectsErr:     true,
			expectedErrMsg: "invalid PCIRoot format",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			testSysfsRoot, err := os.MkdirTemp("", "sysfs_test")
			if err != nil {
				t.Fatalf("Failed to create temp dir: %v", err)
			}
			t.Cleanup(func() {
				if err := os.RemoveAll(testSysfsRoot); err != nil {
					t.Errorf("Failed to remove temp sysfs directory: %v", err)
				}
			})

			ret, err := address.ResolvePCIRoot(test.sysfs(t, testSysfsRoot))

			if test.expectsErr {
				if err == nil {
					t.Errorf("Expected error but got none")
					return
				}
				if !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Errorf("Expected error message %q contains %q", err.Error(), test.expectedErrMsg)
					return
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}
			if ret == nil {
				t.Errorf("Expected non-nil PCIRoot, got nil")
				return
			}
			if ret.String() != root.String() {
				t.Errorf("Expected PCIRoot %s, got %s", root.String(), ret.String())
				return
			}
		})
	}
}
