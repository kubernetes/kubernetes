/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io/ioutil"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/gce_pd"
	"k8s.io/kubernetes/pkg/volume/host_path"
)

func newTestHost(t *testing.T, fakeKubeClient client.Interface) volume.VolumeHost {
	tempDir, err := ioutil.TempDir("/tmp", "persistent_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return volume.NewFakeVolumeHost(tempDir, fakeKubeClient, testProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, ProbeVolumePlugins()))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/persistent-claim" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{}}}}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestNewBuilder(t *testing.T) {
	tests := []struct {
		pv              *api.PersistentVolume
		claim           *api.PersistentVolumeClaim
		plugin          volume.VolumePlugin
		podVolume       api.VolumeSource
		testFunc        func(builder volume.Builder, plugin volume.VolumePlugin) error
		expectedFailure bool
	}{
		{
			pv: &api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name: "pvA",
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
					},
					ClaimRef: &api.ObjectReference{
						Name: "claimA",
					},
				},
			},
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claimA",
					Namespace: "nsA",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "pvA",
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase: api.ClaimBound,
				},
			},
			podVolume: api.VolumeSource{
				PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
					ReadOnly:  false,
					ClaimName: "claimA",
				},
			},
			plugin: gce_pd.ProbeVolumePlugins()[0],
			testFunc: func(builder volume.Builder, plugin volume.VolumePlugin) error {
				if !strings.Contains(builder.GetPath(), util.EscapeQualifiedNameForDisk(plugin.Name())) {
					return fmt.Errorf("builder path expected to contain plugin name.  Got: %s", builder.GetPath())
				}
				return nil
			},
			expectedFailure: false,
		},
		{
			pv: &api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name: "pvB",
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{Path: "/tmp"},
					},
					ClaimRef: &api.ObjectReference{
						Name: "claimB",
					},
				},
			},
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claimB",
					Namespace: "nsB",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "pvA",
				},
			},
			podVolume: api.VolumeSource{
				PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
					ReadOnly:  false,
					ClaimName: "claimB",
				},
			},
			plugin: host_path.ProbeVolumePlugins(volume.VolumeConfig{})[0],
			testFunc: func(builder volume.Builder, plugin volume.VolumePlugin) error {
				if builder.GetPath() != "/tmp" {
					return fmt.Errorf("Expected HostPath.Path /tmp, got: %s", builder.GetPath())
				}
				return nil
			},
			expectedFailure: false,
		},
		{
			pv: &api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name: "pvA",
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
					},
				},
			},
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claimA",
					Namespace: "nsA",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "pvA",
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase: api.ClaimBound,
				},
			},
			podVolume: api.VolumeSource{
				PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
					ReadOnly:  false,
					ClaimName: "claimA",
				},
			},
			plugin: gce_pd.ProbeVolumePlugins()[0],
			testFunc: func(builder volume.Builder, plugin volume.VolumePlugin) error {
				if builder != nil {
					return fmt.Errorf("Unexpected non-nil builder: %+v", builder)
				}
				return nil
			},
			expectedFailure: true, // missing pv.Spec.ClaimRef
		},
		{
			pv: &api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name: "pvA",
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
					},
					ClaimRef: &api.ObjectReference{
						Name: "claimB",
						UID:  types.UID("abc123"),
					},
				},
			},
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claimA",
					Namespace: "nsA",
					UID:       types.UID("def456"),
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "pvA",
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase: api.ClaimBound,
				},
			},
			podVolume: api.VolumeSource{
				PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
					ReadOnly:  false,
					ClaimName: "claimA",
				},
			},
			plugin: gce_pd.ProbeVolumePlugins()[0],
			testFunc: func(builder volume.Builder, plugin volume.VolumePlugin) error {
				if builder != nil {
					return fmt.Errorf("Unexpected non-nil builder: %+v", builder)
				}
				return nil
			},
			expectedFailure: true, // mismatched pv.Spec.ClaimRef and pvc
		},
	}

	for _, item := range tests {
		o := testclient.NewObjects(api.Scheme, api.Scheme)
		o.Add(item.pv)
		o.Add(item.claim)
		client := &testclient.Fake{}
		client.AddReactor("*", "*", testclient.ObjectReaction(o, api.RESTMapper))

		plugMgr := volume.VolumePluginMgr{}
		plugMgr.InitPlugins(testProbeVolumePlugins(), newTestHost(t, client))

		plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
		if err != nil {
			t.Errorf("Can't find the plugin by name")
		}
		spec := &volume.Spec{Volume: &api.Volume{VolumeSource: item.podVolume}}
		pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
		builder, err := plug.NewBuilder(spec, pod, volume.VolumeOptions{}, nil)

		if !item.expectedFailure {
			if err != nil {
				t.Errorf("Failed to make a new Builder: %v", err)
			}
			if builder == nil {
				t.Errorf("Got a nil Builder: %v", builder)
			}
		}

		if err := item.testFunc(builder, item.plugin); err != nil {
			t.Errorf("Unexpected error %+v", err)
		}
	}
}

func TestNewBuilderClaimNotBound(t *testing.T) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvC",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
			},
		},
	}
	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claimC",
			Namespace: "nsA",
		},
	}
	podVolume := api.VolumeSource{
		PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
			ReadOnly:  false,
			ClaimName: "claimC",
		},
	}
	o := testclient.NewObjects(api.Scheme, api.Scheme)
	o.Add(pv)
	o.Add(claim)
	client := &testclient.Fake{}
	client.AddReactor("*", "*", testclient.ObjectReaction(o, api.RESTMapper))

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(testProbeVolumePlugins(), newTestHost(t, client))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &volume.Spec{Volume: &api.Volume{VolumeSource: podVolume}}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.NewBuilder(spec, pod, volume.VolumeOptions{}, nil)
	if builder != nil {
		t.Errorf("Expected a nil builder if the claim wasn't bound")
	}
}

func testProbeVolumePlugins() []volume.VolumePlugin {
	allPlugins := []volume.VolumePlugin{}
	allPlugins = append(allPlugins, gce_pd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, host_path.ProbeVolumePlugins(volume.VolumeConfig{})...)
	allPlugins = append(allPlugins, ProbeVolumePlugins()...)
	return allPlugins
}
