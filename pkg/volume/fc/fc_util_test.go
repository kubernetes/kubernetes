/*
Copyright 2015 The Kubernetes Authors.

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

package fc

import (
	"os"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/volume/util"
)

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

type fakeIOHandler struct{}

func (handler *fakeIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	switch dirname {
	case "/dev/disk/by-path/":
		f1 := &fakeFileInfo{
			name: "pci-0000:41:00.0-fc-0x500a0981891b8dc5-lun-0",
		}
		f2 := &fakeFileInfo{
			name: "fc-0x5005076810213b32-lun-2",
		}
		f3 := &fakeFileInfo{
			name: "abc-0000:41:00.0-fc-0x5005076810213404-lun-0",
		}
		return []os.FileInfo{f1, f2, f3}, nil
	case "/sys/block/":
		f := &fakeFileInfo{
			name: "dm-1",
		}
		return []os.FileInfo{f}, nil
	case "/dev/disk/by-id/":
		f := &fakeFileInfo{
			name: "scsi-3600508b400105e210000900000490000",
		}
		return []os.FileInfo{f}, nil
	}
	return nil, nil
}

func (handler *fakeIOHandler) Lstat(name string) (os.FileInfo, error) {
	return nil, nil
}

func (handler *fakeIOHandler) EvalSymlinks(path string) (string, error) {
	return "/dev/sda", nil
}

func (handler *fakeIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return nil
}

func TestSearchDisk(t *testing.T) {
	tests := []struct {
		name        string
		wwns        []string
		lun         string
		expectError bool
	}{
		{
			name: "PCI disk",
			wwns: []string{"500a0981891b8dc5"},
			lun:  "0",
		},
		{
			name: "Non PCI disk",
			wwns: []string{"5005076810213b32"},
			lun:  "2",
		},
		{
			name:        "Invalid Storage Controller",
			wwns:        []string{"5005076810213404"},
			lun:         "0",
			expectError: true,
		},
		{
			name:        "Non existing disk",
			wwns:        []string{"500507681fffffff"},
			lun:         "0",
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeMounter := fcDiskMounter{
				fcDisk: &fcDisk{
					wwns: test.wwns,
					lun:  test.lun,
					io:   &fakeIOHandler{},
				},
				deviceUtil: util.NewDeviceHandler(util.NewIOHandler()),
			}
			devicePath, err := searchDisk(fakeMounter)
			if test.expectError && err == nil {
				t.Errorf("expected error but got none")
			}
			if !test.expectError && err != nil {
				t.Errorf("got unexpected error: %s", err)
			}
			// if no disk matches input wwn and lun, exit
			if devicePath == "" && !test.expectError {
				t.Errorf("no fc disk found")
			}
		})
	}
}

func TestSearchDiskWWID(t *testing.T) {
	fakeMounter := fcDiskMounter{
		fcDisk: &fcDisk{
			wwids: []string{"3600508b400105e210000900000490000"},
			io:    &fakeIOHandler{},
		},
		deviceUtil: util.NewDeviceHandler(util.NewIOHandler()),
	}
	devicePath, error := searchDisk(fakeMounter)
	// if no disk matches input wwid, exit
	if devicePath == "" || error != nil {
		t.Errorf("no fc disk found")
	}
}

func TestParsePDName(t *testing.T) {
	tests := []struct {
		name        string
		path        string
		wwns        []string
		lun         int32
		wwids       []string
		expectError bool
	}{
		{
			name:  "single WWID",
			path:  "/var/lib/kubelet/plugins/kubernetes.io/fc/60050763008084e6e0000000000001ae",
			wwids: []string{"60050763008084e6e0000000000001ae"},
		},
		{
			name:  "multiple WWID",
			path:  "/var/lib/kubelet/plugins/kubernetes.io/fc/60050763008084e6e0000000000001ae-60050763008084e6e0000000000001af",
			wwids: []string{"60050763008084e6e0000000000001ae", "60050763008084e6e0000000000001af"},
		},
		{
			name: "single WWN",
			path: "/var/lib/kubelet/plugins/kubernetes.io/fc/50050768030539b6-lun-0",
			wwns: []string{"50050768030539b6"},
			lun:  0,
		},
		{
			name: "multiple WWNs",
			path: "/var/lib/kubelet/plugins/kubernetes.io/fc/50050768030539b6-50050768030539b7-lun-0",
			wwns: []string{"50050768030539b6", "50050768030539b7"},
			lun:  0,
		},
		{
			name:        "no WWNs",
			path:        "/var/lib/kubelet/plugins/kubernetes.io/fc/lun-0",
			expectError: true,
		},
		{
			name:        "invalid lun",
			path:        "/var/lib/kubelet/plugins/kubernetes.io/fc/50050768030539b6-lun-x",
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wwns, lun, wwids, err := parsePDName(test.path)
			if test.expectError && err == nil {
				t.Errorf("expected error but got none")
			}
			if !test.expectError && err != nil {
				t.Errorf("got unexpected error: %s", err)
			}
			if !reflect.DeepEqual(wwns, test.wwns) {
				t.Errorf("expected WWNs %+v, got %+v", test.wwns, wwns)
			}
			if lun != test.lun {
				t.Errorf("expected lun %d, got %d", test.lun, lun)
			}
			if !reflect.DeepEqual(wwids, test.wwids) {
				t.Errorf("expected WWIDs %+v, got %+v", test.wwids, wwids)
			}
		})
	}
}
