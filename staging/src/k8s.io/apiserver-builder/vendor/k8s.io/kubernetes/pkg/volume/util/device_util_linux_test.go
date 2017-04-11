// +build linux

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
	"testing"
	"time"
)

type mockOsIOHandler struct{}

func (handler *mockOsIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	switch dirname {
	case "/sys/block/dm-2/slaves/":
		f := &fakeFileInfo{
			name: "sda",
		}
		return []os.FileInfo{f}, nil
	case "/sys/block/":
		f1 := &fakeFileInfo{
			name: "sda",
		}
		f2 := &fakeFileInfo{
			name: "dm-1",
		}
		return []os.FileInfo{f1, f2}, nil
	}
	return nil, nil
}

func (handler *mockOsIOHandler) Lstat(name string) (os.FileInfo, error) {
	links := map[string]string{
		"/sys/block/dm-1/slaves/sda": "sda",
		"/dev/sda":                   "sda",
	}
	if dev, ok := links[name]; ok {
		return &fakeFileInfo{name: dev}, nil
	}
	return nil, errors.New("Not Implemented for Mock")
}

func (handler *mockOsIOHandler) EvalSymlinks(path string) (string, error) {
	links := map[string]string{
		"/returns/a/dev":                                              "/dev/sde",
		"/returns/non/dev":                                            "/sys/block",
		"/dev/disk/by-path/127.0.0.1:3260-eui.02004567A425678D-lun-0": "/dev/sda",
		"/dev/dm-2": "/dev/dm-2",
		"/dev/dm-3": "/dev/dm-3",
		"/dev/sde":  "/dev/sde",
	}
	return links[path], nil
}

func (handler *mockOsIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return errors.New("Not Implemented for Mock")
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
	mockDeviceUtil := NewDeviceHandler(&mockOsIOHandler{})
	dev := mockDeviceUtil.FindMultipathDeviceForDevice("/dev/disk/by-path/127.0.0.1:3260-eui.02004567A425678D-lun-0")
	if dev != "/dev/dm-1" {
		t.Fatalf("mpio device not found dm-1 expected got [%s]", dev)
	}
	dev = mockDeviceUtil.FindMultipathDeviceForDevice("/dev/disk/by-path/empty")
	if dev != "" {
		t.Fatalf("mpio device not found '' expected got [%s]", dev)
	}
}

func TestFindDeviceForPath(t *testing.T) {
	io := &mockOsIOHandler{}

	disk, err := findDeviceForPath("/dev/sde", io)
	if disk != "sde" {
		t.Fatalf("disk [%s] didn't match expected sde", disk)
	}
	disk, err = findDeviceForPath("/returns/a/dev", io)
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
