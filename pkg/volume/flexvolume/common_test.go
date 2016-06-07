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
	"encoding/json"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/exec"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
)

func testPlugin() (*flexVolumePlugin, string) {
	rootDir, err := utiltesting.MkTmpdir("flexvolume_test")
	if err != nil {
		panic("error creating temp dir: " + err.Error())
	}
	return &flexVolumePlugin{
		driverName:          "test",
		execPath:            "/plugin",
		host:                volumetesting.NewFakeVolumeHost(rootDir, nil, nil, ""),
		unsupportedCommands: []string{},
	}, rootDir
}

func assertDriverCall(t *testing.T, output exec.FakeCombinedOutputAction, expectedCommand string, expectedArgs ...string) exec.FakeCommandAction {
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
		return &exec.FakeCmd{
			Argv:                 args,
			CombinedOutputScript: []exec.FakeCombinedOutputAction{output},
		}
	}
}

func fakeRunner(fakeCommands ...exec.FakeCommandAction) exec.Interface {
	return &exec.FakeExec{
		CommandScript: fakeCommands,
	}
}

func fakeResultOutput(result interface{}) exec.FakeCombinedOutputAction {
	return func() ([]byte, error) {
		bytes, err := json.Marshal(result)
		if err != nil {
			panic("Unable to marshal result: " + err.Error())
		}
		return bytes, nil
	}
}

func successOutput() exec.FakeCombinedOutputAction {
	return fakeResultOutput(&DriverStatus{StatusSuccess, "", "", ""})
}

func notSupportedOutput() exec.FakeCombinedOutputAction {
	return fakeResultOutput(&DriverStatus{StatusNotSupported, "", "", ""})
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
	vol := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			FlexVolume: &api.FlexVolumeSource{
				Driver:   "kubernetes.io/fakeAttacher",
				ReadOnly: false,
			},
		},
	}
	return volume.NewSpecFromVolume(vol)
}

func fakePersistentVolumeSpec() *volume.Spec {
	vol := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "vol1",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				FlexVolume: &api.FlexVolumeSource{
					Driver:   "kubernetes.io/fakeAttacher",
					ReadOnly: false,
				},
			},
		},
	}
	return volume.NewSpecFromPersistentVolume(vol, false)
}

func specJson(plugin *flexVolumePlugin, spec *volume.Spec, extraOptions map[string]string) string {
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
