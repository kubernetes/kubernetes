/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package utils

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	dratesting "k8s.io/dynamic-resource-allocation/testing"
)

func TestNewPCIAddress(t *testing.T) {
	tests := map[string]struct {
		domain, bus, device, function uint16
		expectsErr                    bool
		expectedErrMsg                string
	}{
		"valid zero address":    {0, 0, 0, 0, false, ""},
		"valid simple address":  {1, 2, 3, 4, false, ""},
		"valid complex address": {0x1234, 0x56, 0x1e, 0x5, false, ""},
		// no invalid domain case because domain uses full 16 bits
		"invalid bus":      {0, PCIBusMax + 1, 0, 0, true, "invalid PCI bus number"},
		"invalid device":   {0, 0, PCIDeviceMax + 1, 0, true, "invalid PCI device number"},
		"invalid function": {0, 0, 0, PCIFunctionMax + 1, true, "invalid PCI function number"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			address, err := NewPCIAddress(test.domain, test.bus, test.device, test.function)
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
			if address.domain != test.domain || address.bus != test.bus || address.device != test.device || address.function != test.function {
				t.Errorf("Expected PCIAddress(%d, %d, %d, %d), got (%d, %d, %d, %d)",
					test.domain, test.bus, test.device, test.function,
					address.domain, address.bus, address.device, address.function)
				return
			}
		})
	}
}

func TestPCIAddressString(t *testing.T) {
	tests := map[string]struct {
		address  *PCIAddress
		expected string
	}{
		"zero":    {MustNewPCIAddress(0, 0, 0, 0), "0000:00:00.0"},
		"simple":  {MustNewPCIAddress(0x0001, 0x02, 0x03, 0x4), "0001:02:03.4"},
		"complex": {MustNewPCIAddress(0x1234, 0x56, 0x1e, 0x5), "1234:56:1e.5"},
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
		input          string
		expected       *PCIAddress
		expectsErr     bool
		expectedErrMsg string
	}{
		"valid zero":         {"0000:00:00.0", MustNewPCIAddress(0x0, 0x0, 0x0, 0x0), false, ""},
		"valid simple":       {"0001:02:03.4", MustNewPCIAddress(0x1, 0x2, 0x3, 0x4), false, ""},
		"valid complex":      {"1234:56:1e.7", MustNewPCIAddress(0x1234, 0x56, 0x1e, 0x7), false, ""},
		"invalid format":     {"0000-00-00-0", nil, true, "invalid PCI address forma"},
		"too less parts":     {"0000:00:00", nil, true, "invalid PCI address forma"},
		"too many parts":     {"0000:00:00.0.1", nil, true, "invalid PCI address forma"},
		"invalid hex value":  {"0000:00:00.0g", nil, true, "invalid PCI address format"},
		"completely invalid": {"invalid", nil, true, "invalid PCI address forma"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			result, err := ParsePCIAddress(test.input)
			if test.expectsErr {
				if err == nil {
					t.Errorf("Expected error but got none for input %s", test.input)
					return
				}
				if test.expectedErrMsg != "" && !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Errorf("Expected error message %q contains %q for input %s", err.Error(), test.expectedErrMsg, test.input)
					return
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error for input %s: %v", test.input, err)
				return
			}
			if !reflect.DeepEqual(result, test.expected) {
				t.Errorf("ParsePCIAddress(%s) = %+v, want %+v", test.input, result, test.expected)
				return
			}
		})
	}
}

func TestParsePCIAddressRoundTrip(t *testing.T) {
	tests := map[string]struct {
		addressStr string
		expected   *PCIAddress
	}{
		"zero":    {"0000:00:00.0", MustNewPCIAddress(0x0, 0x0, 0x0, 0x0)},
		"simple":  {"0001:02:03.4", MustNewPCIAddress(0x1, 0x2, 0x3, 0x4)},
		"complex": {"1234:56:1e.5", MustNewPCIAddress(0x1234, 0x56, 0x1e, 0x5)},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			address, err := ParsePCIAddress(test.addressStr)
			if err != nil {
				t.Errorf("ParsePCIAddress(%s) returned error: %v", test.addressStr, err)
				return
			}
			if !reflect.DeepEqual(address, test.expected) {
				t.Errorf("ParsePCIAddress(%s) = %+v, want %+v", test.addressStr, address, test.expected)
				return
			}
			if address.String() != test.expected.String() {
				t.Errorf("Expected %s, got %s", test.expected.String(), address.String())
			}
		})
	}
}

func TestNewPCIRoot(t *testing.T) {
	tests := map[string]struct {
		domain         uint16
		bus            uint16
		address        *PCIRoot
		expectsErr     bool
		expectedErrMsg string
	}{
		"valid zero root":    {0, 0, MustNewPCIRoot(0x0, 0x0), false, ""},
		"valid simple root":  {1, 2, MustNewPCIRoot(0x1, 0x2), false, ""},
		"valid complex root": {0x1234, 0x56, MustNewPCIRoot(0x1234, 0x56), false, ""},
		// no invalid domain case because domain uses full 16 bits
		"invalid bus": {0, PCIBusMax + 1, nil, true, "invalid PCI bus number"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			root, err := NewPCIRoot(test.domain, test.bus)
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
			if root.String() != test.address.String() {
				t.Errorf("Expected PCIRoot %s, got %s", test.address.String(), root.String())
				return
			}
		})
	}
}

func TestPCIRootString(t *testing.T) {
	tests := map[string]struct {
		root     *PCIRoot
		expected string
	}{
		"zero":    {MustNewPCIRoot(0x0, 0x0), "0000:00"},
		"simple":  {MustNewPCIRoot(0x1, 0x2), "0001:02"},
		"complex": {MustNewPCIRoot(0x1234, 0x56), "1234:56"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if got := test.root.String(); got != test.expected {
				t.Errorf("Expected %s, got %s", test.expected, got)
			}
		})
	}
}

func TestParsePCIRoot(t *testing.T) {
	tests := map[string]struct {
		input          string
		expected       *PCIRoot
		expectsErr     bool
		expectedErrMsg string
	}{
		"valid zero":                 {"0000:00", MustNewPCIRoot(0x0, 0x0), false, ""},
		"valid simple":               {"0001:02", MustNewPCIRoot(0x1, 0x2), false, ""},
		"valid complex":              {"1234:56", MustNewPCIRoot(0x1234, 0x56), false, ""},
		"invalid format":             {"0000-00", nil, true, "invalid PCIRoot format"},
		"invalid too less parts":     {"0000", nil, true, "invalid PCIRoot format"},
		"invalid too many parts":     {"0000:00:00", nil, true, "invalid PCIRoot format"},
		"invalid hex value":          {"0000:0g", nil, true, "invalid PCIRoot format"},
		"invalid completely invalid": {"invalid", nil, true, "invalid PCIRoot format"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			result, err := ParsePCIRoot(test.input)
			if test.expectsErr {
				if err == nil {
					t.Errorf("Expected error but got none")
					return
				}
				if test.expectedErrMsg != "" && !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Errorf("Expected error message %q contains %q", err.Error(), test.expectedErrMsg)
					return
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}
			if !reflect.DeepEqual(result, test.expected) {
				t.Errorf("Expected PCIRoot %s, got %s", test.expected.String(), result.String())
				return
			}
		})
	}
}

func TestParsePCIRootRoundTrip(t *testing.T) {
	tests := map[string]struct {
		rootStr  string
		expected *PCIRoot
	}{
		"zero":    {"0000:00", MustNewPCIRoot(0x0, 0x0)},
		"simple":  {"0001:02", MustNewPCIRoot(0x1, 0x2)},
		"complex": {"1234:56", MustNewPCIRoot(0x1234, 0x56)},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			root, err := ParsePCIRoot(test.rootStr)
			if err != nil {
				t.Errorf("ParsePCIRoot(%s) returned error: %v", test.rootStr, err)
				return
			}
			if !reflect.DeepEqual(root, test.expected) {
				t.Errorf("ParsePCIRoot(%s) = %+v, want %+v", test.rootStr, root, test.expected)
				return
			}
			if root.String() != test.expected.String() {
				t.Errorf("Expected %s, got %s", test.expected.String(), root.String())
			}
		})
	}
}

func TestPCIAddressResolvePCIRoot(t *testing.T) {
	address := MustNewPCIAddress(0x1234, 0x56, 0x1e, 0x7)
	root := MustNewPCIRoot(0x2345, 0x67)

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
