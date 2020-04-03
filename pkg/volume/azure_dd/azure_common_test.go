// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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

package azure_dd

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/stretchr/testify/assert"
	"k8s.io/utils/exec"
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

var (
	lun       = 1
	lunStr    = "1"
	diskPath  = "4:0:0:" + lunStr
	devName   = "sdd"
	lunStr1   = "2"
	diskPath1 = "3:0:0:" + lunStr1
	devName1  = "sde"
)

type fakeIOHandler struct{}

func (handler *fakeIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	switch dirname {
	case "/sys/bus/scsi/devices":
		f1 := &fakeFileInfo{
			name: "3:0:0:1",
		}
		f2 := &fakeFileInfo{
			name: "4:0:0:0",
		}
		f3 := &fakeFileInfo{
			name: diskPath,
		}
		f4 := &fakeFileInfo{
			name: "host1",
		}
		f5 := &fakeFileInfo{
			name: "target2:0:0",
		}
		return []os.FileInfo{f1, f2, f3, f4, f5}, nil
	case "/sys/bus/scsi/devices/" + diskPath + "/block":
		n := &fakeFileInfo{
			name: devName,
		}
		return []os.FileInfo{n}, nil
	case "/sys/bus/scsi/devices/" + diskPath1 + "/block":
		n := &fakeFileInfo{
			name: devName1,
		}
		return []os.FileInfo{n}, nil
	}
	return nil, fmt.Errorf("bad dir")
}

func (handler *fakeIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return nil
}

func (handler *fakeIOHandler) Readlink(name string) (string, error) {
	return "/dev/azure/disk/sda", nil
}

func (handler *fakeIOHandler) ReadFile(filename string) ([]byte, error) {
	if strings.HasSuffix(filename, "vendor") {
		return []byte("Msft    \n"), nil
	}
	if strings.HasSuffix(filename, "model") {
		return []byte("Virtual Disk \n"), nil
	}
	return nil, fmt.Errorf("unknown file")
}

func TestIoHandler(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		t.Skipf("TestIoHandler not supported on GOOS=%s", runtime.GOOS)
	}
	disk, err := findDiskByLun(lun, &fakeIOHandler{}, exec.New())
	if runtime.GOOS == "windows" {
		if err != nil {
			t.Errorf("no data disk found: disk %v err %v", disk, err)
		}
	} else {
		// if no disk matches lun, exit
		if disk != "/dev/"+devName || err != nil {
			t.Errorf("no data disk found: disk %v err %v", disk, err)
		}
	}
}

func TestNormalizeStorageAccountType(t *testing.T) {
	tests := []struct {
		storageAccountType  string
		expectedAccountType compute.DiskStorageAccountTypes
		expectError         bool
	}{
		{
			storageAccountType:  "",
			expectedAccountType: compute.StandardSSDLRS,
			expectError:         false,
		},
		{
			storageAccountType:  "NOT_EXISTING",
			expectedAccountType: "",
			expectError:         true,
		},
		{
			storageAccountType:  "Standard_LRS",
			expectedAccountType: compute.StandardLRS,
			expectError:         false,
		},
		{
			storageAccountType:  "Premium_LRS",
			expectedAccountType: compute.PremiumLRS,
			expectError:         false,
		},
		{
			storageAccountType:  "StandardSSD_LRS",
			expectedAccountType: compute.StandardSSDLRS,
			expectError:         false,
		},
		{
			storageAccountType:  "UltraSSD_LRS",
			expectedAccountType: compute.UltraSSDLRS,
			expectError:         false,
		},
	}

	for _, test := range tests {
		result, err := normalizeStorageAccountType(test.storageAccountType)
		assert.Equal(t, result, test.expectedAccountType)
		assert.Equal(t, err != nil, test.expectError, fmt.Sprintf("error msg: %v", err))
	}
}

func TestGetDiskNum(t *testing.T) {
	tests := []struct {
		deviceInfo  string
		expectedNum string
		expectError bool
	}{
		{
			deviceInfo:  "/dev/disk0",
			expectedNum: "0",
			expectError: false,
		},
		{
			deviceInfo:  "/dev/disk99",
			expectedNum: "99",
			expectError: false,
		},
		{
			deviceInfo:  "",
			expectedNum: "",
			expectError: true,
		},
		{
			deviceInfo:  "/dev/disk",
			expectedNum: "",
			expectError: true,
		},
		{
			deviceInfo:  "999",
			expectedNum: "",
			expectError: true,
		},
	}

	for _, test := range tests {
		result, err := getDiskNum(test.deviceInfo)
		assert.Equal(t, result, test.expectedNum)
		assert.Equal(t, err != nil, test.expectError, fmt.Sprintf("error msg: %v", err))
	}
}
