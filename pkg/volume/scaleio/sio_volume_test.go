/*
Copyright 2015 The Kubernetes Authors.

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

package scaleio

import (
	"fmt"
	"os"
	"path"
	"testing"

	api "k8s.io/kubernetes/pkg/api/v1"
	fakeclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

var (
	testSioSystem  = "sio"
	testSioPD      = "default"
	testSioVol     = "vol-0001"
	testSioVolName = fmt.Sprintf("%s.%s.%s", testSioSystem, testSioPD, testSioVol)
)

func newPluginMgr(t *testing.T) (*volume.VolumePluginMgr, string) {
	tmpDir, err := utiltesting.MkTmpdir("scaleio-test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	config := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      "sio-secret",
			Namespace: "default",
			UID:       "1234567890",
		},
		Data: map[string][]byte{
			"username": []byte("username"),
			"password": []byte("password"),
		},
	}
	fakeClient := fakeclient.NewSimpleClientset(config)
	host := volumetest.NewFakeVolumeHost(tmpDir, fakeClient, nil)
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), host)
	return plugMgr, tmpDir
}

func TestVolumeCanSupport(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t)
	defer os.RemoveAll(tmpDir)
	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %s by name", sioPluginName)
	}
	if plug.GetPluginName() != "kubernetes.io/scaleio" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(
		&volume.Spec{
			Volume: &api.Volume{
				VolumeSource: api.VolumeSource{
					ScaleIO: &api.ScaleIOVolumeSource{},
				},
			},
		},
	) {
		t.Errorf("Expected true for CanSupport LibStorage VolumeSource")
	}
	if !plug.CanSupport(
		&volume.Spec{
			PersistentVolume: &api.PersistentVolume{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						ScaleIO: &api.ScaleIOVolumeSource{},
					},
				},
			},
		},
	) {
		t.Errorf("Expected true for CanSupport LibStorage PersistentVolumeSource")
	}
}

func TestVolumeGetAccessModes(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t)
	defer os.RemoveAll(tmpDir)
	plug, err := plugMgr.FindPersistentPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", sioPluginName)
	}
	if !containsMode(plug.GetAccessModes(), api.ReadWriteOnce) {
		t.Errorf("Expected two AccessModeTypes:  %s or %s", api.ReadWriteOnce, api.ReadOnlyMany)
	}
}
func containsMode(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func TestVolumeMounterUnmounter(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t)
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Errorf("Cannot assert plugin to be type sioPlugin")
	}

	sioPlug.mounter = &mount.FakeMounter{}
	vol := &api.Volume{
		Name: "vol-0001",
		VolumeSource: api.VolumeSource{
			ScaleIO: &api.ScaleIOVolumeSource{
				Gateway:          "http://test.scaleio:1111",
				System:           testSioSystem,
				ProtectionDomain: testSioPD,
				StoragePool:      "default",
				VolumeName:       testSioVolName,
				FSType:           "ext4",
				SecretRef:        &api.LocalObjectReference{Name: "sio-secret"},
			},
		},
	}

	sioMounter, err := sioPlug.NewMounter(
		volume.NewSpecFromVolume(vol),
		&api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if sioMounter == nil {
		t.Fatal("Got a nil Mounter")
	}
	sio := newFakeSio()
	sioMounter.(*sioVolume).sioMgr.client = sio
	sioMounter.(*sioVolume).sioMgr.CreateVolume(testSioVolName, 8) //create vol ahead of time

	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~scaleio/"+testSioVolName)
	path := sioMounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := sioMounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	// unmount
	sioUnmounter, err := sioPlug.NewUnmounter(testSioVolName, types.UID("poduid"))
	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}
	if sioUnmounter == nil {
		t.Fatal("Got a nil Unmounter")
	}
	sioUnmounter.(*sioVolume).sioMgr.client = sio

	if err := sioUnmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}

func TestVolumeProvisioner(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t)
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Errorf("Cannot assert plugin to be type sioPlugin")
	}

	options := volume.VolumeOptions{
		ClusterName: "testcluster",
		PVName:      "siopvc",
		PVC:         volumetest.CreateTestPVC("100Mi", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}

	// incomplete options, test should fail
	_, err = sioPlug.NewProvisioner(options)
	if err == nil {
		t.Fatal("expected failure due to incomplete options")
	}

	options.Parameters = map[string]string{
		confKey.gateway:          "http://test.scaleio:11111",
		confKey.system:           "sio",
		confKey.protectionDomain: testSioPD,
		confKey.storagePool:      "default",
		confKey.secretRef:        "sio-secret",
	}
	provisioner, err := sioPlug.NewProvisioner(options)
	if err != nil {
		t.Fatalf("failed to create new provisioner: %v", err)
	}
	if provisioner == nil {
		t.Fatal("got a nil provisioner")
	}
	sio := newFakeSio()
	provisioner.(*sioVolume).sioMgr.client = sio

	spec, err := provisioner.Provision()
	if err != nil {
		t.Fatalf("call to Provision() failed: %v", err)
	}
	expectedSpecName := options.ClusterName + "-dynamic-" + options.PVName
	actualSpecName := spec.Spec.PersistentVolumeSource.ScaleIO.VolumeName
	if actualSpecName != expectedSpecName {
		t.Errorf("expecting volume name %s, got %s", expectedSpecName, actualSpecName)
	}

	vol, err := sio.FindVolume(actualSpecName)
	if err != nil {
		t.Fatalf("failed getting volume: %v", err)
	}
	if vol.Name != expectedSpecName {
		t.Errorf("expected volume name to be %s, got %s", expectedSpecName, vol.Name)
	}
}
