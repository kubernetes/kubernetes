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
	"encoding/json"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/test/utils/harness"
	"k8s.io/utils/exec"
	exectesting "k8s.io/utils/exec/testing"
)

func testPlugin(h *harness.Harness) (*flexVolumeAttachablePlugin, string) {
	rootDir := h.TempDir("", "flexvolume_test")
	return &flexVolumeAttachablePlugin{
		flexVolumePlugin: &flexVolumePlugin{
			driverName:          "test",
			execPath:            "/plugin",
			host:                volumetesting.NewFakeVolumeHost(h.T, rootDir, nil, nil),
			unsupportedCommands: []string{},
		},
	}, rootDir
}

func assertDriverCall(t *harness.Harness, output exectesting.FakeAction, expectedCommand string, expectedArgs ...string) exectesting.FakeCommandAction {
	return func(cmd string, args ...string) exec.Cmd {
		if cmd != "/plugin/test" {
			t.Errorf("Wrong executable called: got %v, expected %v", cmd, "/plugin/test")
		}
		if args[0] != expectedCommand {
			t.Errorf("Wrong command called: got %v, expected %v", args[0], expectedCommand)
		}
		cmdArgs := args[1:]
		if !sameArgs(cmdArgs, expectedArgs) {
			t.Errorf("Wrong args for %s: got %v, expected %v", args[0], cmdArgs, expectedArgs)
		}
		return &exectesting.FakeCmd{
			Argv:                 args,
			CombinedOutputScript: []exectesting.FakeAction{output},
		}
	}
}

func fakeRunner(fakeCommands ...exectesting.FakeCommandAction) exec.Interface {
	return &exectesting.FakeExec{
		CommandScript: fakeCommands,
	}
}

func fakeResultOutput(result interface{}) exectesting.FakeAction {
	return func() ([]byte, []byte, error) {
		bytes, err := json.Marshal(result)
		if err != nil {
			panic("Unable to marshal result: " + err.Error())
		}
		return bytes, nil, nil
	}
}

func successOutput() exectesting.FakeAction {
	return fakeResultOutput(&DriverStatus{StatusSuccess, "", "", "", true, nil, 0})
}

func notSupportedOutput() exectesting.FakeAction {
	return fakeResultOutput(&DriverStatus{StatusNotSupported, "", "", "", false, nil, 0})
}

func sameArgs(args, expectedArgs []string) bool {
	if len(args) != len(expectedArgs) {
		return false
	}
	for i, v := range args {
		if v != expectedArgs[i] {
			return false
		}
	}
	return true
}

func fakeVolumeSpec() *volume.Spec {
	vol := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			FlexVolume: &v1.FlexVolumeSource{
				Driver:   "kubernetes.io/fakeAttacher",
				ReadOnly: false,
			},
		},
	}
	return volume.NewSpecFromVolume(vol)
}

func specJSON(plugin *flexVolumeAttachablePlugin, spec *volume.Spec, extraOptions map[string]string) string {
	o, err := NewOptionsForDriver(spec, plugin.host, extraOptions)
	if err != nil {
		panic("Failed to convert spec: " + err.Error())
	}
	bytes, err := json.Marshal(o)
	if err != nil {
		panic("Unable to marshal result: " + err.Error())
	}
	return string(bytes)
}
