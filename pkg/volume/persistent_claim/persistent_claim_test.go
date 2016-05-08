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
	"os"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/types"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/gce_pd"
	"k8s.io/kubernetes/pkg/volume/host_path"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// newTestHost returns the temp directory and the VolumeHost created.
// Please be sure to cleanup the temp directory once done!
func newTestHost(t *testing.T, clientset clientset.Interface) (string, volume.VolumeHost) {
	tempDir, err := utiltesting.MkTmpdir("persistent_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return tempDir, volumetest.NewFakeVolumeHost(tempDir, clientset, testProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost("/somepath/fake", nil, ProbeVolumePlugins()))

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

func TestNewMounter(t *testing.T) {
	tests := []struct {
		pv              *api.PersistentVolume
		claim           *api.PersistentVolumeClaim
		plugin          volume.VolumePlugin
		podVolume       api.VolumeSource
		testFunc        func(mounter volume.Mounter, plugin volume.VolumePlugin, pod *api.Pod) error
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
			testFunc: func(mounter volume.Mounter, plugin volume.VolumePlugin, pod *api.Pod) error {
				if !strings.Contains(mounter.GetPath(), utilstrings.EscapeQualifiedNameForDisk(plugin.Name())) {
					return fmt.Errorf("mounter path expected to contain plugin name.  Got: %s", mounter.GetPath())
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
						HostPath: &api.HostPathVolumeSource{Path: "/somepath"},
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
			testFunc: func(mounter volume.Mounter, plugin volume.VolumePlugin, pod *api.Pod) error {
				if mounter.GetPath() != "/somepath" {
					return fmt.Errorf("Expected HostPath.Path /somepath, got: %s", mounter.GetPath())
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
			testFunc: func(mounter volume.Mounter, plugin volume.VolumePlugin, pod *api.Pod) error {
				if mounter != nil {
					return fmt.Errorf("Unexpected non-nil mounter: %+v", mounter)
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
			testFunc: func(mounter volume.Mounter, plugin volume.VolumePlugin, pod *api.Pod) error {
				if mounter != nil {
					return fmt.Errorf("Unexpected non-nil mounter: %+v", mounter)
				}
				return nil
			},
			expectedFailure: true, // mismatched pv.Spec.ClaimRef and pvc
		},
		{ // Test GID annotation
			pv: &api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name: "pv",
					Annotations: map[string]string{
						volumeGidAnnotationKey: "12345",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
					},
					ClaimRef: &api.ObjectReference{
						Name: "claim",
						UID:  types.UID("abc123"),
					},
				},
			},
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name: "claim",
					UID:  types.UID("abc123"),
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "pv",
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase: api.ClaimBound,
				},
			},
			podVolume: api.VolumeSource{
				PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
					ReadOnly:  false,
					ClaimName: "claim",
				},
			},
			plugin: gce_pd.ProbeVolumePlugins()[0],
			testFunc: func(mounter volume.Mounter, plugin volume.VolumePlugin, pod *api.Pod) error {
				if pod.Spec.SecurityContext == nil {
					return fmt.Errorf("Pod SecurityContext was not set")
				}

				if pod.Spec.SecurityContext.SupplementalGroups[0] != 12345 {
					return fmt.Errorf("Pod's SupplementalGroups list does not contain expect group")
				}

				return nil
			},
			expectedFailure: false,
		},
	}

	for _, item := range tests {
		client := fake.NewSimpleClientset(item.pv, item.claim)

		plugMgr := volume.VolumePluginMgr{}
		tempDir, vh := newTestHost(t, client)
		defer os.RemoveAll(tempDir)
		plugMgr.InitPlugins(testProbeVolumePlugins(), vh)

		plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
		if err != nil {
			t.Errorf("Can't find the plugin by name")
		}
		spec := &volume.Spec{Volume: &api.Volume{VolumeSource: item.podVolume}}
		pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
		mounter, err := plug.NewMounter(spec, pod, volume.VolumeOptions{})

		if !item.expectedFailure {
			if err != nil {
				t.Errorf("Failed to make a new Mounter: %v", err)
			}
			if mounter == nil {
				t.Errorf("Got a nil Mounter: %v", mounter)
			}
		}

		if err := item.testFunc(mounter, item.plugin, pod); err != nil {
			t.Errorf("Unexpected error %+v", err)
		}
	}
}

func TestNewMounterClaimNotBound(t *testing.T) {
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
	client := fake.NewSimpleClientset(pv, claim)

	plugMgr := volume.VolumePluginMgr{}
	tempDir, vh := newTestHost(t, client)
	defer os.RemoveAll(tempDir)
	plugMgr.InitPlugins(testProbeVolumePlugins(), vh)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/persistent-claim")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &volume.Spec{Volume: &api.Volume{VolumeSource: podVolume}}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(spec, pod, volume.VolumeOptions{})
	if mounter != nil {
		t.Errorf("Expected a nil mounter if the claim wasn't bound")
	}
}

func testProbeVolumePlugins() []volume.VolumePlugin {
	allPlugins := []volume.VolumePlugin{}
	allPlugins = append(allPlugins, gce_pd.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, host_path.ProbeVolumePlugins(volume.VolumeConfig{})...)
	allPlugins = append(allPlugins, ProbeVolumePlugins()...)
	return allPlugins
}
