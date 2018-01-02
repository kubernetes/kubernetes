// Copyright 2017 Google Inc. All Rights Reserved.
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
package accelerators

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/mindprince/gonvml"
	"github.com/stretchr/testify/assert"
)

func updateFile(t *testing.T, fn string, content []byte) {
	if err := ioutil.WriteFile(fn, content, 0666); err != nil {
		t.Fatalf("Error writing to temporary file for testing: %v", err)
	}
}

func TestDetectDevices(t *testing.T) {
	sysFsPCIDevicesPath = "/non-existent-path"
	detected := detectDevices("0x10de")
	assert.False(t, detected)

	var err error
	// Create temporary directory to represent sysfs pci devices path
	if sysFsPCIDevicesPath, err = ioutil.TempDir("", "sys-bus-pci-devices"); err != nil {
		t.Fatalf("Error creating temporary directory for testing: %v", err)
	}
	defer os.RemoveAll(sysFsPCIDevicesPath)

	device0 := filepath.Join(sysFsPCIDevicesPath, "device0")
	device1 := filepath.Join(sysFsPCIDevicesPath, "device1")
	device2 := filepath.Join(sysFsPCIDevicesPath, "device2")
	for _, device := range []string{device0, device1, device2} {
		if err = os.Mkdir(device, 0777); err != nil {
			t.Fatalf("Error creating temporary directory for testing: %v", err)
		}
	}

	// device0 directory is present to make sure that
	// we handle bad device directories case correctly.

	// A valid vendor file but different than what's being detected.
	updateFile(t, filepath.Join(device1, "vendor"), []byte("0x8086\n"))
	detected = detectDevices("0x10de")
	assert.False(t, detected)

	// vendor file for device being detected
	updateFile(t, filepath.Join(device2, "vendor"), []byte("0x10de\n"))
	detected = detectDevices("0x10de")
	assert.True(t, detected)
}

func TestGetCollector(t *testing.T) {
	// Mock parseDevicesCgroup.
	originalParser := parseDevicesCgroup
	mockParser := func(_ string) ([]int, error) {
		return []int{2, 3}, nil
	}
	parseDevicesCgroup = mockParser
	defer func() {
		parseDevicesCgroup = originalParser
	}()

	nm := &NvidiaManager{}

	// When nvmlInitialized is false, empty collector should be returned.
	ac, err := nm.GetCollector("does-not-matter")
	assert.Nil(t, err)
	assert.NotNil(t, ac)
	nc, ok := ac.(*NvidiaCollector)
	assert.True(t, ok)
	assert.Equal(t, 0, len(nc.Devices))

	// When nvidiaDevices is empty, empty collector should be returned.
	nm.nvmlInitialized = true
	ac, err = nm.GetCollector("does-not-matter")
	assert.Nil(t, err)
	assert.NotNil(t, ac)
	nc, ok = ac.(*NvidiaCollector)
	assert.True(t, ok)
	assert.Equal(t, 0, len(nc.Devices))

	// nvidiaDevices contains devices but they are different than what
	// is returned by parseDevicesCgroup. We should get an error.
	nm.nvidiaDevices = map[int]gonvml.Device{0: {}, 1: {}}
	ac, err = nm.GetCollector("does-not-matter")
	assert.NotNil(t, err)
	assert.NotNil(t, ac)
	nc, ok = ac.(*NvidiaCollector)
	assert.True(t, ok)
	assert.Equal(t, 0, len(nc.Devices))

	// nvidiaDevices contains devices returned by parseDevicesCgroup.
	// No error should be returned and collectors devices array should be
	// correctly initialized.
	nm.nvidiaDevices[2] = gonvml.Device{}
	nm.nvidiaDevices[3] = gonvml.Device{}
	ac, err = nm.GetCollector("does-not-matter")
	assert.Nil(t, err)
	assert.NotNil(t, ac)
	nc, ok = ac.(*NvidiaCollector)
	assert.True(t, ok)
	assert.Equal(t, 2, len(nc.Devices))
}

func TestParseDevicesCgroup(t *testing.T) {
	// Test case for empty devices cgroup path
	nvidiaMinorNumbers, err := parseDevicesCgroup("")
	assert.NotNil(t, err)
	assert.Equal(t, []int{}, nvidiaMinorNumbers)

	// Test case for non-existent devices cgroup
	nvidiaMinorNumbers, err = parseDevicesCgroup("/non-existent-path")
	assert.NotNil(t, err)
	assert.Equal(t, []int{}, nvidiaMinorNumbers)

	// Create temporary directory to represent devices cgroup.
	tmpDir, err := ioutil.TempDir("", "devices-cgroup")
	if err != nil {
		t.Fatalf("Error creating temporary directory for testing: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	tmpfn := filepath.Join(tmpDir, "devices.list")

	// Test case when devices.list file has more than three fields.
	updateFile(t, tmpfn, []byte("c 1:2 rwm badformat\n"))
	nvidiaMinorNumbers, err = parseDevicesCgroup(tmpDir)
	assert.NotNil(t, err)
	assert.Equal(t, []int{}, nvidiaMinorNumbers)

	// Test case when devices.list file's second field is not major:minor.
	updateFile(t, tmpfn, []byte("c badformat rwm\n"))
	nvidiaMinorNumbers, err = parseDevicesCgroup(tmpDir)
	assert.NotNil(t, err)
	assert.Equal(t, []int{}, nvidiaMinorNumbers)

	// Test case with nvidia devices present
	updateFile(t, tmpfn, []byte("c 195:0 rwm\nc 195:255 rwm\nc 195:1 rwm"))
	nvidiaMinorNumbers, err = parseDevicesCgroup(tmpDir)
	assert.Nil(t, err)
	assert.Equal(t, []int{0, 1}, nvidiaMinorNumbers) // Note that 255 is not supposed to be returned.

	// Test case with a common devices.list file
	updateFile(t, tmpfn, []byte("a *:* rwm\n"))
	nvidiaMinorNumbers, err = parseDevicesCgroup(tmpDir)
	assert.Nil(t, err)
	assert.Equal(t, []int{}, nvidiaMinorNumbers)

	// Test case for empty devices.list file
	updateFile(t, tmpfn, []byte(""))
	nvidiaMinorNumbers, err = parseDevicesCgroup(tmpDir)
	assert.Nil(t, err)
	assert.Equal(t, []int{}, nvidiaMinorNumbers)
}
