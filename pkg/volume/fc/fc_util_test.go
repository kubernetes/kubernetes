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
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	testingexec "k8s.io/utils/exec/testing"

	volumetest "k8s.io/kubernetes/pkg/volume/testing"
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
		f4 := &fakeFileInfo{
			name: "pci-0000:41:00.0-fc-0x500a0981891b8dc5-lun-12",
		}
		f5 := &fakeFileInfo{
			name: "pci-0000:41:00.0-fc-0x500a0981891b8dc5-lun-1",
		}
		f6 := &fakeFileInfo{
			name: "fc-0x5005076810213b32-lun-25",
		}
		f7 := &fakeFileInfo{
			name: "fc-0x500507681021a537-lun-0",
		}
		f8 := &fakeFileInfo{
			name: "fc-0x500507681022a554-lun-2",
		}
		return []os.FileInfo{f4, f5, f6, f1, f2, f3, f7, f8}, nil
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
	links := map[string]string{
		"/sys/block/dm-1/slaves/sde": "sde",
		"/sys/block/dm-1/slaves/sdf": "sdf",
		"/sys/block/dm-1/slaves/sdg": "sdg",
	}
	if dev, ok := links[name]; ok {
		return &fakeFileInfo{name: dev}, nil
	}
	return nil, errors.New("device not found for mock")
}

func (handler *fakeIOHandler) EvalSymlinks(path string) (string, error) {
	switch path {
	case "/dev/disk/by-path/pci-0000:41:00.0-fc-0x500a0981891b8dc5-lun-0":
		return "/dev/sda", nil
	case "/dev/disk/by-path/pci-0000:41:00.0-fc-0x500a0981891b8dc5-lun-1":
		return "/dev/sdb", nil
	case "/dev/disk/by-path/fc-0x5005076810213b32-lun-2":
		return "/dev/sdc", nil
	case "/dev/disk/by-path/pci-0000:41:00.0-fc-0x500a0981891b8dc5-lun-12":
		return "/dev/sdl", nil
	case "/dev/disk/by-path/fc-0x5005076810213b32-lun-25":
		return "/dev/sdx", nil
	case "/dev/disk/by-id/scsi-3600508b400105e210000900000490000":
		return "/dev/sdd", nil
	case "/dev/disk/by-path/fc-0x500507681021a537-lun-0":
		return "/dev/sde", nil
	case "/dev/disk/by-path/fc-0x500507681022a554-lun-2":
		return "/dev/sdf", nil
	case "/dev/sde":
		return "/dev/sde", nil
	case "/dev/sdf":
		return "/dev/sdf", nil
	}
	return "", nil
}

func (handler *fakeIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return nil
}

func (handler *fakeIOHandler) ReadFile(filename string) ([]byte, error) {
	return nil, nil
}

type fakeDetachIOHandler struct {
	fakeIOHandler
	t                *testing.T
	linkTarget       string
	expectedEvalPath string
	evalErr          error
	mapPath          string
	writtenFiles     []string
}

func (handler *fakeDetachIOHandler) EvalSymlinks(path string) (string, error) {
	handler.t.Helper()
	if path != handler.expectedEvalPath {
		handler.t.Errorf("evaluated symlink path = %q, want %q", path, handler.expectedEvalPath)
	}
	if handler.evalErr != nil {
		return "", handler.evalErr
	}
	return handler.linkTarget, nil
}

func (handler *fakeDetachIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	handler.t.Helper()
	if _, err := os.Stat(handler.mapPath); err != nil {
		handler.t.Errorf("global map path was removed before device cleanup: %v", err)
	}
	handler.writtenFiles = append(handler.writtenFiles, filename)
	return nil
}

type fakeDetachDeviceUtil struct {
	util.DeviceUtil
	multipathDevice string
	slaveDevices    []string
}

func (handler *fakeDetachDeviceUtil) FindMultipathDeviceForDevice(disk string) string {
	return handler.multipathDevice
}

func (handler *fakeDetachDeviceUtil) FindSlaveDevicesOnMultipath(disk string) []string {
	return handler.slaveDevices
}

type fakeDetachManager struct {
	diskManager
	readDir func(string) ([]os.DirEntry, error)
}

func (manager *fakeDetachManager) DetachBlockFCDisk(c fcDiskUnmapper, mapPath, devicePath string) error {
	return (&fcUtil{}).detachBlockFCDisk(c, mapPath, devicePath, manager.readDir)
}

func TestTearDownDeviceRecoversFromStaleDevicePath(t *testing.T) {
	const wwn = "50050768030539b6"
	const wwid = "3600508b400105e210000900000490000"
	tests := []struct {
		name          string
		volumeInfo    string
		devicePath    string
		linkName      string
		linkTarget    string
		readDirErr    error
		evalErr       error
		multipath     string
		slaveDevices  []string
		cleanupErr    bool
		wantSearchDir string
		wantWrites    []string
		wantErr       bool
	}{
		{
			name:          "WWN and LUN identity resolves current device",
			volumeInfo:    wwn + "-lun-0",
			linkName:      "pci-0000:41:00.0-fc-0x" + wwn + "-lun-0",
			linkTarget:    "/dev/sdy",
			wantSearchDir: byPath,
			wantWrites:    []string{"/sys/block/sdy/device/delete"},
		},
		{
			name:          "WWID identity resolves current device",
			volumeInfo:    wwid,
			linkName:      "scsi-" + wwid,
			linkTarget:    "/dev/sdz",
			wantSearchDir: byID,
			wantWrites:    []string{"/sys/block/sdz/device/delete"},
		},
		{
			name:          "identity resolves multipath device",
			volumeInfo:    wwid,
			linkName:      "scsi-" + wwid,
			linkTarget:    "/dev/sda",
			multipath:     "/dev/dm-1",
			slaveDevices:  []string{"/dev/sda", "/dev/sdb"},
			wantSearchDir: byID,
			wantWrites: []string{
				"/sys/block/sda/device/delete",
				"/sys/block/sdb/device/delete",
			},
		},
		{
			name:          "stale path and identity no longer resolves",
			volumeInfo:    wwid,
			wantSearchDir: byID,
		},
		{
			name:          "identity directory no longer exists",
			volumeInfo:    wwid,
			readDirErr:    os.ErrNotExist,
			wantSearchDir: byID,
		},
		{
			name:          "identity disappears while resolving",
			volumeInfo:    wwid,
			linkName:      "scsi-" + wwid,
			evalErr:       os.ErrNotExist,
			wantSearchDir: byID,
		},
		{
			name:          "empty path and identity resolves",
			volumeInfo:    wwn + "-lun-0",
			devicePath:    "empty",
			linkName:      "fc-0x" + wwn + "-lun-0",
			linkTarget:    "/dev/sdc",
			wantSearchDir: byPath,
			wantWrites:    []string{"/sys/block/sdc/device/delete"},
		},
		{
			name:          "empty path without identity still returns error",
			volumeInfo:    wwn + "-lun-0",
			devicePath:    "empty",
			wantSearchDir: byPath,
			wantErr:       true,
		},
		{
			name:          "existing path remains normal",
			volumeInfo:    wwn + "-lun-0",
			devicePath:    "existing",
			linkTarget:    "/dev/sdd",
			wantSearchDir: byPath,
			wantWrites:    []string{"/sys/block/sdd/device/delete"},
		},
		{
			name:          "identity lookup access error is returned",
			volumeInfo:    wwid,
			readDirErr:    os.ErrPermission,
			wantSearchDir: byID,
			wantErr:       true,
		},
		{
			name:          "identity resolution access error is returned",
			volumeInfo:    wwid,
			linkName:      "scsi-" + wwid,
			evalErr:       os.ErrPermission,
			wantSearchDir: byID,
			wantErr:       true,
		},
		{
			name:          "multipath cleanup error is returned",
			volumeInfo:    wwid,
			linkName:      "scsi-" + wwid,
			linkTarget:    "/dev/sda",
			multipath:     "/dev/dm-1",
			cleanupErr:    true,
			wantSearchDir: byID,
			wantErr:       true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mapPath := filepath.Join(t.TempDir(), test.volumeInfo)
			if err := os.Mkdir(mapPath, 0750); err != nil {
				t.Fatalf("failed to create global map path: %v", err)
			}
			devicePath := filepath.Join(t.TempDir(), "missing-device")
			if test.devicePath == "empty" {
				devicePath = ""
			} else if test.devicePath == "existing" {
				if err := os.WriteFile(devicePath, nil, 0600); err != nil {
					t.Fatalf("failed to create remembered device path: %v", err)
				}
			}
			var readDirs []string
			readDir := func(dirname string) ([]os.DirEntry, error) {
				readDirs = append(readDirs, dirname)
				if test.readDirErr != nil {
					return nil, test.readDirErr
				}
				entries := []os.DirEntry{
					fs.FileInfoToDirEntry(&fakeFileInfo{name: "scsi-unrelated-volume"}),
				}
				if test.linkName != "" {
					entries = append(entries, fs.FileInfoToDirEntry(&fakeFileInfo{name: test.linkName}))
				}
				return entries, nil
			}
			expectedEvalPath := ""
			if test.linkName != "" {
				expectedEvalPath = filepath.Join(test.wantSearchDir, test.linkName)
			} else if test.devicePath == "existing" {
				expectedEvalPath = devicePath
			}
			io := &fakeDetachIOHandler{
				t:                t,
				linkTarget:       test.linkTarget,
				expectedEvalPath: expectedEvalPath,
				evalErr:          test.evalErr,
				mapPath:          mapPath,
			}
			manager := &fakeDetachManager{readDir: readDir}
			scripts := []volumetest.CommandScript{}
			if test.multipath != "" {
				returnCode := 0
				if test.cleanupErr {
					returnCode = 1
				}
				scripts = append(scripts, volumetest.CommandScript{
					Cmd: "multipath", Args: []string{"-f", test.multipath}, ReturnCode: returnCode,
				})
			}
			if !test.cleanupErr {
				for _, write := range test.wantWrites {
					device := strings.TrimSuffix(strings.TrimPrefix(write, "/sys/block/"), "/device/delete")
					scripts = append(scripts, volumetest.CommandScript{
						Cmd: "blockdev", Args: []string{"--flushbufs", "/dev/" + device},
					})
				}
			}
			fakeExec := &testingexec.FakeExec{ExactOrder: true}
			if len(scripts) > 0 {
				volumetest.ScriptCommands(fakeExec, scripts)
			} else {
				fakeExec.DisableScripts = true
			}
			unmapper := &fcDiskUnmapper{
				fcDisk: &fcDisk{
					manager: manager,
					io:      io,
				},
				deviceUtil: &fakeDetachDeviceUtil{
					multipathDevice: test.multipath,
					slaveDevices:    test.slaveDevices,
				},
				exec: fakeExec,
			}

			err := unmapper.TearDownDevice(mapPath, devicePath)
			if (err != nil) != test.wantErr {
				t.Fatalf("TearDownDevice() error = %v, wantErr %t", err, test.wantErr)
			}
			if test.wantSearchDir == "" {
				if len(readDirs) != 0 {
					t.Errorf("unexpected identity searches: %v", readDirs)
				}
			} else if !reflect.DeepEqual(readDirs, []string{test.wantSearchDir}) {
				t.Errorf("search directories = %v, want %q", readDirs, test.wantSearchDir)
			}
			if fakeExec.CommandCalls != len(scripts) {
				t.Errorf("executed commands = %d, want %d", fakeExec.CommandCalls, len(scripts))
			}
			if !reflect.DeepEqual(io.writtenFiles, test.wantWrites) {
				t.Errorf("device cleanup writes = %v, want %v", io.writtenFiles, test.wantWrites)
			}
			_, statErr := os.Stat(mapPath)
			if test.wantErr && statErr != nil {
				t.Errorf("expected global map path to remain after error, stat error: %v", statErr)
			} else if !test.wantErr && !os.IsNotExist(statErr) {
				t.Errorf("expected global map path to be removed, stat error: %v", statErr)
			}
		})
	}
}

func TestSearchDisk(t *testing.T) {
	tests := []struct {
		name        string
		wwns        []string
		lun         string
		disk        string
		expectError bool
	}{
		{
			name: "PCI disk 0",
			wwns: []string{"500a0981891b8dc5"},
			lun:  "0",
			disk: "/dev/sda",
		},
		{
			name: "PCI disk 1",
			wwns: []string{"500a0981891b8dc5"},
			lun:  "1",
			disk: "/dev/sdb",
		},
		{
			name: "Non PCI disk",
			wwns: []string{"5005076810213b32"},
			lun:  "2",
			disk: "/dev/sdc",
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
				deviceUtil: util.NewDeviceHandler(&fakeIOHandler{}),
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
			if devicePath != test.disk {
				t.Errorf("matching wrong disk, expected: %s, actual: %s", test.disk, devicePath)
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
