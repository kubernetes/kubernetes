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

	"k8s.io/mount-utils"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/test/utils/harness"
)

func TestSetUpAt(tt *testing.T) {
	t := harness.For(tt)
	defer t.Close()

	spec := fakeVolumeSpec()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-pod",
			Namespace: "my-ns",
			UID:       types.UID("my-uid"),
		},
		Spec: v1.PodSpec{
			ServiceAccountName: "my-sa",
		},
	}
	mounter := mount.NewFakeMounter(nil)

	plugin, rootDir := testPlugin(t)
	plugin.unsupportedCommands = []string{"unsupportedCmd"}
	plugin.runner = fakeRunner(
		// first call without mounterArgs.FsGroup
		assertDriverCall(t, successOutput(), mountCmd, rootDir+"/mount-dir",
			specJSON(plugin, spec, map[string]string{
				optionKeyPodName:            "my-pod",
				optionKeyPodNamespace:       "my-ns",
				optionKeyPodUID:             "my-uid",
				optionKeyServiceAccountName: "my-sa",
			})),

		// second test has mounterArgs.FsGroup
		assertDriverCall(t, notSupportedOutput(), mountCmd, rootDir+"/mount-dir",
			specJSON(plugin, spec, map[string]string{
				optionFSGroup:               "42",
				optionKeyPodName:            "my-pod",
				optionKeyPodNamespace:       "my-ns",
				optionKeyPodUID:             "my-uid",
				optionKeyServiceAccountName: "my-sa",
			})),
		assertDriverCall(t, fakeVolumeNameOutput("sdx"), getVolumeNameCmd,
			specJSON(plugin, spec, nil)),
	)

	m, _ := plugin.newMounterInternal(spec, pod, mounter, plugin.runner)
	var mounterArgs volume.MounterArgs
	m.SetUpAt(rootDir+"/mount-dir", mounterArgs)

	group := int64(42)
	mounterArgs.FsGroup = &group
	m.SetUpAt(rootDir+"/mount-dir", mounterArgs)
}
