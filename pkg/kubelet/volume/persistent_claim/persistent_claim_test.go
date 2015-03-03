/*
Copyright 2014 Google Inc. All rights reserved.

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

package persistent_claim

import (
	"io/ioutil"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

func newTestHost(t *testing.T, fakeKubeClient client.Interface) volume.Host {
	tempDir, err := ioutil.TempDir("/tmp", "persistent_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}

	return &volume.FakeHost{tempDir, fakeKubeClient}
}

func TestCanSupport(t *testing.T) {
	plugMgr := volume.PluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), &volume.FakeHost{"/tmp/fake", nil})

	plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/persistent-claim" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&api.Volume{Source: api.VolumeSource{HostPath: &api.HostPathVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&api.Volume{Source: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{}}}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&api.Volume{Source: api.VolumeSource{}}) {
		t.Errorf("Expected false")
	}
}

func TestNewBuilder(t *testing.T) {

	tests := []struct {
		pv        api.PersistentVolume
		claim     api.PersistentVolumeClaim
		podVolume api.VolumeSource
	}{
		{
			pv: api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name: "pvA",
				},
				Spec: api.PersistentVolumeSpec{
					Source: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
					},
				},
				ClaimRef: &api.ObjectReference{
					Name: "claimA",
				},
			},
			claim: api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claimA",
					Namespace: "nsA",
				},
				Status: api.PersistentVolumeClaimStatus{
					VolumeRef: &api.ObjectReference{
						Name: "pvA",
					},
				},
			},
			podVolume: api.VolumeSource{
				PersistentVolumeClaimVolumeSource: &api.PersistentVolumeClaimVolumeSource{
					AccessMode: api.ReadWriteOnce,
					PersistentVolumeClaimRef: &api.ObjectReference{
						Name:      "claimA",
						Namespace: "nsA",
					},
				},
			},
		},
	}

	for _, item := range tests {

		client := &client.Fake{
			PersistentVolume:      item.pv,
			PersistentVolumeClaim: item.claim,
		}

		plugMgr := volume.PluginMgr{}
		plugMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, client))

		plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
		if err != nil {
			t.Errorf("Can't find the plugin by name")
		}
		spec := &api.Volume{
			Name:   "vol1",
			Source: item.podVolume,
		}
		builder, err := plug.NewBuilder(spec, types.UID("poduid"))
		if err != nil {
			t.Errorf("Failed to make a new Builder: %v", err)
		}
		if builder == nil {
			t.Errorf("Got a nil Builder: %v")
		}
	}

}

//
//func TestPlugin(t *testing.T) {
//	plugMgr := volume.PluginMgr{}
//	plugMgr.InitPlugins(ProbeVolumePlugins(), &volume.FakeHost{"/tmp/fake", nil})
//
//	plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
//	if err != nil {
//		t.Errorf("Can't find the plugin by name")
//	}
//	spec := &api.Volume{
//		Name:   "vol1",
//		Source: api.VolumeSource{HostPath: &api.HostPathVolumeSource{}},
//	}
//	builder, err := plug.NewBuilder(spec, types.UID("poduid"))
//	if err != nil {
//		t.Errorf("Failed to make a new Builder: %v", err)
//	}
//	if builder == nil {
//		t.Errorf("Got a nil Builder: %v")
//	}
//
//	path := builder.GetPath()
//	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~persistent-claim/vol1" {
//		t.Errorf("Got unexpected path: %s", path)
//	}
//
//	if err := builder.SetUp(); err != nil {
//		t.Errorf("Expected success, got: %v", err)
//	}
//	if _, err := os.Stat(path); err != nil {
//		if os.IsNotExist(err) {
//			t.Errorf("SetUp() failed, volume path not created: %s", path)
//		} else {
//			t.Errorf("SetUp() failed: %v", err)
//		}
//	}
//
//	cleaner, err := plug.NewCleaner("vol1", types.UID("poduid"))
//	if err != nil {
//		t.Errorf("Failed to make a new Cleaner: %v", err)
//	}
//	if cleaner == nil {
//		t.Errorf("Got a nil Cleaner: %v")
//	}
//
//	if err := cleaner.TearDown(); err != nil {
//		t.Errorf("Expected success, got: %v", err)
//	}
//	if _, err := os.Stat(path); err == nil {
//		t.Errorf("TearDown() failed, volume path still exists: %s", path)
//	} else if !os.IsNotExist(err) {
//		t.Errorf("SetUp() failed: %v", err)
//	}
//}
