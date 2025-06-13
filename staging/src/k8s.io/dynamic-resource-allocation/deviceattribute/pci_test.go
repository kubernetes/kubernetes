package deviceattribute

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/utils/ptr"

	resourceapi "k8s.io/dynamic-resource-allocation/api"
	dratesting "k8s.io/dynamic-resource-allocation/testing"
	utils "k8s.io/dynamic-resource-allocation/utils"
)

func TestStandardPCIDeviceAttributes(t *testing.T) {
	pcieRootAttrName := resourceapi.QualifiedName(resourceapi.StandardDeviceAttributePCIeRoot)

	address := &utils.PCIAddress{Domain: 0, Bus: 1, Device: 2, Function: 3}
	root := &utils.PCIRoot{Domain: 0, Bus: 1}
	standardAttributes := map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		pcieRootAttrName: {
			StringValue: ptr.To("pci" + root.String()),
		},
	}

	tests := map[string]struct {
		pciAddress         *utils.PCIAddress
		sysfs              func(*testing.T, string) utils.Sysfs
		expectedAttributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
		expectsErr         bool
		expectedErrMsg     string
	}{
		"nil address": {
			pciAddress: nil,
			sysfs: func(t *testing.T, testSysfsRoot string) utils.Sysfs {
				return utils.NewSysfs()
			},
			expectedAttributes: nil,
		},
		"valid address": {
			pciAddress: address,
			sysfs: func(t *testing.T, testSysfsRoot string) utils.Sysfs {
				sysfs := utils.NewSysfsWithRoot(testSysfsRoot)
				devicePath := sysfs.Devices(filepath.Join("pci"+root.String(), address.String()))
				dratesting.TouchFile(t, devicePath)

				busPath := sysfs.Bus(filepath.Join("pci", "devices", address.String()))
				dratesting.CreateSymlink(t, devicePath, busPath)
				return sysfs
			},
			expectedAttributes: standardAttributes,
		},
		"non-existent address": {
			pciAddress: address,
			sysfs: func(t *testing.T, testSysfsRoot string) utils.Sysfs {
				return utils.NewSysfsWithRoot(testSysfsRoot)
			},
			expectsErr:     true,
			expectedErrMsg: "no such file or directory",
		},
		"devices path not a symlink": {
			pciAddress: address,
			sysfs: func(t *testing.T, testSysfsRoot string) utils.Sysfs {
				sysfs := utils.NewSysfsWithRoot(testSysfsRoot)
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
			pciAddress: address,
			sysfs: func(t *testing.T, testSysfsRoot string) utils.Sysfs {
				sysfs := utils.NewSysfsWithRoot(testSysfsRoot)
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
			pciAddress: address,
			sysfs: func(t *testing.T, testSysfsRoot string) utils.Sysfs {
				sysfs := utils.NewSysfsWithRoot(testSysfsRoot)
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
			testSysfsRoot, err := os.MkdirTemp("", "sysfs-test")
			if err != nil {
				t.Fatalf("Failed to create temp sysfs directory: %v", err)
			}
			t.Cleanup(func() {
				if err := os.RemoveAll(testSysfsRoot); err != nil {
					t.Errorf("Failed to remove temp sysfs directory: %v", err)
				}
			})
			result, err := StandardPCIDeviceAttributes(
				WithPCIDeviceAddress(test.pciAddress),
				withSysfs(test.sysfs(t, testSysfsRoot)),
			)
			if test.expectsErr {
				if err == nil {
					t.Fatalf("Expected error but got none")
				}
				if !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Fatalf("Expected error message %q contains %q", err.Error(), test.expectedErrMsg)
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if !reflect.DeepEqual(result, test.expectedAttributes) {
				t.Fatalf("Expected attributes %v, got %v", test.expectedAttributes, result)
			}
		})
	}
}
