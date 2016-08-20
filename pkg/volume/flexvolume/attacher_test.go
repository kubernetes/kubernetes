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

package flexvolume

import (
	"testing"
	"time"
)

func TestAttach(t *testing.T) {
	spec := fakeVolumeSpec()

	plugin, _ := testPlugin()
	plugin.runner = fakeRunner(
		assertDriverCall(t, notSupportedOutput(), attachCmd,
			specJson(plugin, spec, nil), "localhost"),
	)

	a, _ := plugin.NewAttacher()
	a.Attach(spec, "localhost")
}

func TestWaitForAttach(t *testing.T) {
	spec := fakeVolumeSpec()

	plugin, _ := testPlugin()
	plugin.runner = fakeRunner(
		assertDriverCall(t, notSupportedOutput(), waitForAttachCmd,
			specJson(plugin, spec, nil), "/dev/sdx"),
	)

	a, _ := plugin.NewAttacher()
	a.WaitForAttach(spec, "/dev/sdx", 1*time.Second)
}

func TestGetDeviceMountPath(t *testing.T) {
	spec := fakeVolumeSpec()

	plugin, rootDir := testPlugin()
	mountsDir := rootDir + "/plugins/kubernetes.io/flexvolume/test/mounts"
	extraOptions := map[string]string{
		optionMountsDir: mountsDir,
	}
	plugin.runner = fakeRunner(
		assertDriverCall(t, notSupportedOutput(), getDeviceMountPathCmd,
			specJson(plugin, spec, extraOptions)),
		// the default is based on plugin.GetDeviceName
		assertDriverCall(t, fakeDeviceNameOutput("sdx"), getVolumeNameCmd,
			specJson(plugin, spec, nil)),
		// second call for the driverSupport cache test
		// (attacher shouldn't call getDeviceMountPathCmd anymore)
		assertDriverCall(t, fakeDeviceNameOutput("sdx"), getVolumeNameCmd,
			specJson(plugin, spec, nil)),
	)

	expectedPath := mountsDir + "/sdx"
	a, _ := plugin.NewAttacher()
	path, err := a.GetDeviceMountPath(spec)
	if err != nil {
		t.Errorf("GetDeviceMountPath() failed: %v", err)
	}
	if path != expectedPath {
		t.Errorf("GetDeviceMountPath() returns %v instead of %v", path, expectedPath)
	}

	path2, err := a.GetDeviceMountPath(spec)
	if err != nil {
		t.Errorf("GetDeviceMountPath() failed: %v", err)
	}
	if path2 != expectedPath {
		t.Errorf("GetDeviceMountPath() returns %v instead of %v", path2, expectedPath)
	}
}

func TestMountDevice(t *testing.T) {
	spec := fakeVolumeSpec()

	plugin, rootDir := testPlugin()
	plugin.runner = fakeRunner(
		assertDriverCall(t, notSupportedOutput(), mountDeviceCmd,
			specJson(plugin, spec, nil), "/dev/sdx", rootDir+"/mount-dir"),
	)

	a, _ := plugin.NewAttacher()
	a.MountDevice(spec, "/dev/sdx", rootDir+"/mount-dir")
}
