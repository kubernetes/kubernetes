//go:build linux

/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"errors"
	"os"
	"reflect"
	"regexp"
	"testing"
	"time"
)

type mockOsIOHandler struct{}

func (handler *mockOsIOHandler) ReadFile(filename string) ([]byte, error) {
	portPattern := regexp.MustCompile("^/sys/class/iscsi_host/(host\\d)/device/session\\d/connection\\d:0/iscsi_connection/connection\\d:0/(?:persistent_)?port$")
	if portPattern.MatchString(filename) {
		return []byte("3260"), nil
	}
	addressPattern := regexp.MustCompile("^/sys/class/iscsi_host/(host\\d)/device/session\\d/connection\\d:0/iscsi_connection/connection\\d:0/(?:persistent_)?address$")
	matches := addressPattern.FindStringSubmatch(filename)
	if nil != matches {
		switch matches[1] {
		case "host2":
			return []byte("10.0.0.1"), nil
		case "host3":
			return []byte("10.0.0.2"), nil
		}
	}
	targetNamePattern := regexp.MustCompile("^/sys/class/iscsi_host/(host\\d)/device/session\\d/iscsi_session/session\\d/targetname$")
	matches = targetNamePattern.FindStringSubmatch(filename)
	if nil != matches {
		switch matches[1] {
		case "host2":
			return []byte("target1"), nil
		case "host3":
			return []byte("target2"), nil
		}
	}
	return nil, errors.New("not Implemented for Mock")
}

func (handler *mockOsIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	switch dirname {
	case "/sys/block/dm-1/slaves":
		f1 := &fakeFileInfo{
			name: "sda",
		}
		f2 := &fakeFileInfo{
			name: "sdb",
		}
		return []os.FileInfo{f1, f2}, nil
	case "/sys/block/":
		f1 := &fakeFileInfo{
			name: "sda",
		}
		f2 := &fakeFileInfo{
			name: "dm-1",
		}
		return []os.FileInfo{f1, f2}, nil
	case "/sys/class/iscsi_host":
		f1 := &fakeFileInfo{
			name: "host2",
		}
		f2 := &fakeFileInfo{
			name: "host3",
		}
		f3 := &fakeFileInfo{
			name: "ignore",
		}
		return []os.FileInfo{f1, f2, f3}, nil
	case "/sys/class/iscsi_host/host2/device":
		f1 := &fakeFileInfo{
			name: "session1",
		}
		f2 := &fakeFileInfo{
			name: "ignore",
		}
		return []os.FileInfo{f1, f2}, nil
	case "/sys/class/iscsi_host/host3/device":
		f1 := &fakeFileInfo{
			name: "session2",
		}
		f2 := &fakeFileInfo{
			name: "ignore",
		}
		return []os.FileInfo{f1, f2}, nil
	case "/sys/class/iscsi_host/host2/device/session1":
		f1 := &fakeFileInfo{
			name: "connection1:0",
		}
		f2 := &fakeFileInfo{
			name: "ignore",
		}
		return []os.FileInfo{f1, f2}, nil
	case "/sys/class/iscsi_host/host3/device/session2":
		f1 := &fakeFileInfo{
			name: "connection2:0",
		}
		f2 := &fakeFileInfo{
			name: "ignore",
		}
		return []os.FileInfo{f1, f2}, nil
	case "/sys/class/iscsi_host/host2/device/session1/target2:0:0/2:0:0:1/block":
		f1 := &fakeFileInfo{
			name: "sda",
		}
		return []os.FileInfo{f1}, nil
	case "/sys/class/iscsi_host/host2/device/session1/target2:0:0/2:0:0:2/block":
		f1 := &fakeFileInfo{
			name: "sdc",
		}
		return []os.FileInfo{f1}, nil
	case "/sys/class/iscsi_host/host3/device/session2/target3:0:0/3:0:0:1/block":
		f1 := &fakeFileInfo{
			name: "sdb",
		}
		return []os.FileInfo{f1}, nil
	case "/sys/class/iscsi_host/host3/device/session2/target3:0:0/3:0:0:2/block":
		f1 := &fakeFileInfo{
			name: "sdd",
		}
		return []os.FileInfo{f1}, nil
	}
	return nil, errors.New("not Implemented for Mock")
}

func (handler *mockOsIOHandler) Lstat(name string) (os.FileInfo, error) {
	links := map[string]string{
		"/sys/block/dm-1/slaves/sda": "sda",
		"/dev/sda":                   "sda",
		"/sys/class/iscsi_host/host2/device/session1/target2:0:0/2:0:0:1": "2:0:0:1",
		"/sys/class/iscsi_host/host2/device/session1/target2:0:0/2:0:0:2": "2:0:0:2",
		"/sys/class/iscsi_host/host3/device/session2/target3:0:0/3:0:0:1": "3:0:0:1",
		"/sys/class/iscsi_host/host3/device/session2/target3:0:0/3:0:0:2": "3:0:0:2",
	}
	if dev, ok := links[name]; ok {
		return &fakeFileInfo{name: dev}, nil
	}
	return nil, errors.New("not Implemented for Mock")
}

func (handler *mockOsIOHandler) EvalSymlinks(path string) (string, error) {
	links := map[string]string{
		"/returns/a/dev":   "/dev/sde",
		"/returns/non/dev": "/sys/block",
		"/dev/disk/by-path/127.0.0.1:3260-eui.02004567A425678D-lun-0": "/dev/sda",
		"/dev/disk/by-path/127.0.0.3:3260-eui.03004567A425678D-lun-0": "/dev/sdb",
		"/dev/dm-2": "/dev/dm-2",
		"/dev/dm-3": "/dev/dm-3",
		"/dev/sdc":  "/dev/sdc",
		"/dev/sde":  "/dev/sde",
	}
	return links[path], nil
}

func (handler *mockOsIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return errors.New("not Implemented for Mock")
}

type fakeFileInfo struct {
	name string
}

func (fi *fakeFileInfo) Name() string {
	return fi.name
}

func (fi *fakeFileInfo) Size() int64 {
	return 0
}

func (fi *fakeFileInfo) Mode() os.FileMode {
	return 777
}

func (fi *fakeFileInfo) ModTime() time.Time {
	return time.Now()
}
func (fi *fakeFileInfo) IsDir() bool {
	return false
}

func (fi *fakeFileInfo) Sys() interface{} {
	return nil
}

func TestFindMultipathDeviceForDevice(t *testing.T) {
	tests := []struct {
		name           string
		device         string
		expectedResult string
	}{
		{
			name:           "Device is already a dm device",
			device:         "/dev/dm-1",
			expectedResult: "/dev/dm-1",
		},
		{
			name:           "Device has no multipath",
			device:         "/dev/sdc",
			expectedResult: "",
		},
		{
			name:           "Device has multipath",
			device:         "/dev/disk/by-path/127.0.0.1:3260-eui.02004567A425678D-lun-0",
			expectedResult: "/dev/dm-1",
		},
		{
			name:           "Invalid device path",
			device:         "/dev/nonexistent",
			expectedResult: "",
		},
	}

	mockDeviceUtil := NewDeviceHandler(&mockOsIOHandler{})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mockDeviceUtil.FindMultipathDeviceForDevice(tt.device)
			if result != tt.expectedResult {
				t.Errorf("FindMultipathDeviceForDevice(%s) = %s, want %s", tt.device, result, tt.expectedResult)
			}
		})
	}
}

func TestFindDeviceForPath(t *testing.T) {
	io := &mockOsIOHandler{}

	disk, err := findDeviceForPath("/dev/sde", io)
	if err != nil {
		t.Fatalf("error finding device for path /dev/sde:%v", err)
	}
	if disk != "sde" {
		t.Fatalf("disk [%s] didn't match expected sde", disk)
	}
	disk, err = findDeviceForPath("/returns/a/dev", io)
	if err != nil {
		t.Fatalf("error finding device for path /returns/a/dev:%v", err)
	}
	if disk != "sde" {
		t.Fatalf("disk [%s] didn't match expected sde", disk)
	}
	_, err = findDeviceForPath("/returns/non/dev", io)
	if err == nil {
		t.Fatalf("link is to incorrect dev")
	}

	_, err = findDeviceForPath("/path/doesnt/exist", &osIOHandler{})
	if err == nil {
		t.Fatalf("path shouldn't exist but still doesn't give an error")
	}

}

func TestFindSlaveDevicesOnMultipath(t *testing.T) {
	mockDeviceUtil := NewDeviceHandler(&mockOsIOHandler{})
	devices := mockDeviceUtil.FindSlaveDevicesOnMultipath("/dev/dm-1")
	if !reflect.DeepEqual(devices, []string{"/dev/sda", "/dev/sdb"}) {
		t.Fatalf("failed to find devices managed by mpio device. /dev/sda, /dev/sdb expected got [%s]", devices)
	}
	dev := mockDeviceUtil.FindSlaveDevicesOnMultipath("/dev/sdc")
	if len(dev) != 0 {
		t.Fatalf("mpio device not found '' expected got [%s]", dev)
	}
}

func TestGetISCSIPortalHostMapForTarget(t *testing.T) {
	mockDeviceUtil := NewDeviceHandler(&mockOsIOHandler{})
	portalHostMap, err := mockDeviceUtil.GetISCSIPortalHostMapForTarget("target1")
	if err != nil {
		t.Fatalf("error getting scsi hosts for target: %v", err)
	}
	if portalHostMap == nil {
		t.Fatal("no portal host map returned")
	}
	if len(portalHostMap) != 1 {
		t.Fatalf("wrong number of map entries in portal host map: %d", len(portalHostMap))
	}
	if portalHostMap["10.0.0.1:3260"] != 2 {
		t.Fatalf("incorrect entry in portal host map: %v", portalHostMap)
	}
}

func TestFindDevicesForISCSILun(t *testing.T) {
	mockDeviceUtil := NewDeviceHandler(&mockOsIOHandler{})
	devices, err := mockDeviceUtil.FindDevicesForISCSILun("target1", 1)
	if err != nil {
		t.Fatalf("error getting devices for lun: %v", err)
	}
	if devices == nil {
		t.Fatal("no devices returned")
	}
	if len(devices) != 1 {
		t.Fatalf("wrong number of devices: %d", len(devices))
	}
	if devices[0] != "sda" {
		t.Fatalf("incorrect device %v", devices)
	}
}
