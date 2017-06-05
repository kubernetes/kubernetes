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

package flexvolume

import (
	"k8s.io/kubernetes/pkg/volume"
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
		assertDriverCall(t, notSupportedOutput(), waitForAttachCmd, "/dev/sdx",
			specJson(plugin, spec, nil)),
	)

	a, _ := plugin.NewAttacher()
	a.WaitForAttach(spec, "/dev/sdx", 1*time.Second)
}

func TestMountDevice(t *testing.T) {
	spec := fakeVolumeSpec()

	plugin, rootDir := testPlugin()
	plugin.runner = fakeRunner(
		assertDriverCall(t, notSupportedOutput(), mountDeviceCmd, rootDir+"/mount-dir", "/dev/sdx",
			specJson(plugin, spec, nil)),
	)

	a, _ := plugin.NewAttacher()
	a.MountDevice(spec, "/dev/sdx", rootDir+"/mount-dir")
}

func TestIsVolumeAttached(t *testing.T) {
	spec := fakeVolumeSpec()

	plugin, _ := testPlugin()
	plugin.runner = fakeRunner(
		assertDriverCall(t, notSupportedOutput(), isAttached, specJson(plugin, spec, nil), "localhost"),
	)
	a, _ := plugin.NewAttacher()
	specs := []*volume.Spec{spec}
	a.VolumesAreAttached(specs, "localhost")
}
