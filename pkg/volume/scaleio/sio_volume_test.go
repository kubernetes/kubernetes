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

package scaleio

import (
	"fmt"
	"os"
	"path"
	"strings"
	"testing"

	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	api "k8s.io/kubernetes/pkg/api/v1"
	fakeclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

var (
	testSioSystem  = "sio"
	testSioPD      = "default"
	testSioVol     = "vol-0001"
	testns         = "default"
	testSioVolName = fmt.Sprintf("%s%s%s", testns, "-", testSioVol)
	podUID         = types.UID("sio-pod")
)

func newPluginMgr(t *testing.T) (*volume.VolumePluginMgr, string) {
	tmpDir, err := utiltesting.MkTmpdir("scaleio-test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	config := &api.Secret{
		ObjectMeta: meta.ObjectMeta{
			Name:      "sio-secret",
			Namespace: testns,
			UID:       "1234567890",
		},
		Type: api.SecretType("kubernetes.io/scaleio"),
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
		Name: testSioVolName,
		VolumeSource: api.VolumeSource{
			ScaleIO: &api.ScaleIOVolumeSource{
				Gateway:          "http://test.scaleio:1111",
				System:           testSioSystem,
				ProtectionDomain: testSioPD,
				StoragePool:      "default",
				VolumeName:       testSioVol,
				FSType:           "ext4",
				SecretRef:        &api.LocalObjectReference{Name: "sio-secret"},
			},
		},
	}

	sioMounter, err := sioPlug.NewMounter(
		volume.NewSpecFromVolume(vol),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if sioMounter == nil {
		t.Fatal("Got a nil Mounter")
	}

	sio := newFakeSio()
	sioVol := sioMounter.(*sioVolume)
	if err := sioVol.setSioMgr(); err != nil {
		t.Fatalf("failed to create sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	sioVol.sioMgr.CreateVolume(testSioVol, 8) //create vol ahead of time

	volPath := path.Join(tmpDir, fmt.Sprintf("pods/%s/volumes/kubernetes.io~scaleio/%s", podUID, testSioVolName))
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

	// rebuild spec
	builtSpec, err := sioPlug.ConstructVolumeSpec(volume.NewSpecFromVolume(vol).Name(), path)
	if err != nil {
		t.Errorf("ConstructVolumeSpec failed %v", err)
	}
	if builtSpec.Name() != vol.Name {
		t.Errorf("Unexpected spec name %s", builtSpec.Name())
	}

	// unmount
	sioUnmounter, err := sioPlug.NewUnmounter(volume.NewSpecFromVolume(vol).Name(), podUID)
	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}
	if sioUnmounter == nil {
		t.Fatal("Got a nil Unmounter")
	}
	sioVol = sioUnmounter.(*sioVolume)
	if err := sioVol.resetSioMgr(); err != nil {
		t.Fatalf("failed to reset sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio

	if err := sioUnmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	// is mount point gone ?
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	// are we still mapped
	if sio.volume.MappedSdcInfo != nil {
		t.Errorf("expected SdcMappedInfo to be nil, volume may still be mapped")
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
		PVName:      "pvc-sio-dynamic-vol",
		PVC:         volumetest.CreateTestPVC("100Mi", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}
	options.PVC.Namespace = testns

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
	sioVol := provisioner.(*sioVolume)
	if err := sioVol.setSioMgrFromConfig(); err != nil {
		t.Fatalf("failed to create scaleio mgr from config: %v", err)
	}
	sioVol.sioMgr.client = sio

	spec, err := provisioner.Provision()
	if err != nil {
		t.Fatalf("call to Provision() failed: %v", err)
	}

	spec.Spec.ClaimRef = &api.ObjectReference{Namespace: testns}

	// validate provision
	actualSpecName := spec.Name
	actualVolName := spec.Spec.PersistentVolumeSource.ScaleIO.VolumeName
	if !strings.HasPrefix(actualSpecName, "pvc-") {
		t.Errorf("expecting volume name to start with pov-, got %s", actualSpecName)
	}

	vol, err := sio.FindVolume(actualVolName)
	if err != nil {
		t.Fatalf("failed getting volume %v: %v", actualVolName, err)
	}
	if vol.Name != actualVolName {
		t.Errorf("expected volume name to be %s, got %s", actualVolName, vol.Name)
	}

	// mount dynamic vol
	sioMounter, err := sioPlug.NewMounter(
		volume.NewSpecFromPersistentVolume(spec, false),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}
	sioVol = sioMounter.(*sioVolume)
	if err := sioVol.setSioMgr(); err != nil {
		t.Fatalf("failed to create sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	if err := sioMounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	// teardown dynamic vol
	sioUnmounter, err := sioPlug.NewUnmounter(spec.Name, podUID)
	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}
	sioVol = sioUnmounter.(*sioVolume)
	if err := sioVol.resetSioMgr(); err != nil {
		t.Fatalf("failed to reset sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	if err := sioUnmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	// test deleter
	deleter, err := sioPlug.NewDeleter(volume.NewSpecFromPersistentVolume(spec, false))
	if err != nil {
		t.Fatalf("failed to create a deleter %v", err)
	}
	sioVol = deleter.(*sioVolume)
	if err := sioVol.setSioMgrFromSpec(); err != nil {
		t.Fatalf("failed to set sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	if err := deleter.Delete(); err != nil {
		t.Fatalf("failed while deleteing vol: %v", err)
	}
	path := deleter.GetPath()
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("Deleter did not delete path %v: %v", path, err)
	}
}
