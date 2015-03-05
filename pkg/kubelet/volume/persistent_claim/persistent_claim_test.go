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
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/host_path"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/gce_pd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"fmt"
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
		plugin	  volume.Plugin
		claim     api.PersistentVolumeClaim
		podVolume api.VolumeSource
		testFunc  func(builder volume.Builder, plugin volume.Plugin) bool
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
			plugin: gce_pd.ProbeVolumePlugins()[0],
			testFunc: func(builder volume.Builder, plugin volume.Plugin) bool {
				if !strings.Contains(builder.GetPath(), volume.EscapePluginName(plugin.Name())){
					t.Errorf("builder path expected to contain plugin name.  Got: %s", builder.GetPath())
					return false
				}
				return true
			},
		},
		{
			pv: api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name: "pvB",
				},
				Spec: api.PersistentVolumeSpec{
					Source: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{ Path: "/tmp"},
					},
				},
				ClaimRef: &api.ObjectReference{
					Name: "claimB",
				},
			},
			claim: api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claimB",
					Namespace: "nsB",
				},
				Status: api.PersistentVolumeClaimStatus{
					VolumeRef: &api.ObjectReference{
						Name: "pvB",
					},
				},
			},
			podVolume: api.VolumeSource{
				PersistentVolumeClaimVolumeSource: &api.PersistentVolumeClaimVolumeSource{
					AccessMode: api.ReadWriteOnce,
					PersistentVolumeClaimRef: &api.ObjectReference{
						Name:      "claimB",
						Namespace: "nsB",
					},
				},
			},
			plugin: host_path.ProbeVolumePlugins()[0],
			testFunc: func(builder volume.Builder, plugin volume.Plugin) bool {
				if builder.GetPath() != "/tmp" {
					t.Errorf("Expected HostPath.Path /tmp, got: %s", builder.GetPath())
					return false
				}
				return true
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

		fmt.Println("builder = " + builder.GetPath())

		if !item.testFunc(builder, item.plugin) {
			t.Errorf("Unexpected errors")
		}
	}
}
