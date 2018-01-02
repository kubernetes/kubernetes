// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package api

import (
	"testing"

	"github.com/google/cadvisor/integration/framework"
)

func TestMachineInformationIsReturned(t *testing.T) {
	fm := framework.New(t)
	defer fm.Cleanup()

	machineInfo, err := fm.Cadvisor().Client().MachineInfo()
	if err != nil {
		t.Fatal(err)
	}

	// Check for "sane" values. Note these can change with time.
	if machineInfo.NumCores <= 0 || machineInfo.NumCores >= 1000000 {
		t.Errorf("Machine info has unexpected number of cores: %v", machineInfo.NumCores)
	}
	if machineInfo.MemoryCapacity <= 0 || machineInfo.MemoryCapacity >= (1<<50 /* 1PB */) {
		t.Errorf("Machine info has unexpected amount of memory: %v", machineInfo.MemoryCapacity)
	}
	if len(machineInfo.Filesystems) == 0 {
		t.Errorf("Expected to have some filesystems, found none")
	}
	for _, fs := range machineInfo.Filesystems {
		if fs.Device == "" {
			t.Errorf("Expected a non-empty device name in: %+v", fs)
		}
		if fs.Capacity < 0 || fs.Capacity >= (1<<60 /* 1 EB*/) {
			t.Errorf("Unexpected capacity in device %q: %v", fs.Device, fs.Capacity)
		}
		if fs.Type == "" {
			t.Errorf("Filesystem type is not set")
		} else if fs.Type == "vfs" && fs.Inodes == 0 {
			t.Errorf("Inodes not available for device %q", fs.Device)
		}
	}
}
