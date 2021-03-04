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
	"testing"

	"k8s.io/kubernetes/test/utils/harness"
	"k8s.io/utils/exec/testing"
)

func TestInit(tt *testing.T) {
	t := harness.For(tt)
	defer t.Close()

	plugin, _ := testPlugin(t)
	plugin.runner = fakeRunner(
		assertDriverCall(t, successOutput(), "init"),
	)
	plugin.Init(plugin.host)
}

func fakeVolumeNameOutput(name string) testingexec.FakeAction {
	return fakeResultOutput(&DriverStatus{
		Status:     StatusSuccess,
		VolumeName: name,
	})
}

func TestGetVolumeName(tt *testing.T) {
	t := harness.For(tt)
	defer t.Close()

	spec := fakeVolumeSpec()
	plugin, _ := testPlugin(t)
	plugin.runner = fakeRunner(
		assertDriverCall(t, fakeVolumeNameOutput(spec.Name()), getVolumeNameCmd,
			specJSON(plugin, spec, nil)),
	)

	name, err := plugin.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetVolumeName() failed: %v", err)
	}
	expectedName := spec.Name()
	if name != expectedName {
		t.Errorf("GetVolumeName() returned %v instead of %v", name, expectedName)
	}
}
