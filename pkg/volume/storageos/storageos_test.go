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

package storageos

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/utils/exec/testing"
	"k8s.io/utils/mount"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("storageos_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/storageos")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/storageos" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{StorageOS: &v1.StorageOSVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{StorageOS: &v1.StorageOSPersistentVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("storageos_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/storageos")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteOnce) || !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", v1.ReadWriteOnce, v1.ReadOnlyMany)
	}
}

type fakePDManager struct {
	api                apiImplementer
	attachCalled       bool
	attachDeviceCalled bool
	detachCalled       bool
	mountCalled        bool
	unmountCalled      bool
	createCalled       bool
	deleteCalled       bool
}

func (fake *fakePDManager) NewAPI(apiCfg *storageosAPIConfig) error {
	fake.api = fakeAPI{}
	return nil
}

func (fake *fakePDManager) CreateVolume(p *storageosProvisioner) (*storageosVolume, error) {
	fake.createCalled = true
	labels := make(map[string]string)
	labels["fakepdmanager"] = "yes"
	return &storageosVolume{
		Name:      "test-storageos-name",
		Namespace: "test-storageos-namespace",
		Pool:      "test-storageos-pool",
		SizeGB:    100,
		Labels:    labels,
		FSType:    "ext2",
	}, nil
}

func (fake *fakePDManager) AttachVolume(b *storageosMounter) (string, error) {
	fake.attachCalled = true
	return "", nil
}

func (fake *fakePDManager) AttachDevice(b *storageosMounter, dir string) error {
	fake.attachDeviceCalled = true
	return nil
}

func (fake *fakePDManager) DetachVolume(b *storageosUnmounter, loopDevice string) error {
	fake.detachCalled = true
	return nil
}

func (fake *fakePDManager) MountVolume(b *storageosMounter, mntDevice, deviceMountPath string) error {
	fake.mountCalled = true
	return nil
}

func (fake *fakePDManager) UnmountVolume(b *storageosUnmounter) error {
	fake.unmountCalled = true
	return nil
}

func (fake *fakePDManager) DeleteVolume(d *storageosDeleter) error {
	fake.deleteCalled = true
	if d.volName != "test-storageos-name" {
		return fmt.Errorf("Deleter got unexpected volume name: %s", d.volName)
	}
	return nil
}

func (fake *fakePDManager) DeviceDir(mounter *storageosMounter) string {
	return defaultDeviceDir
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("storageos_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/storageos")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	secretName := "very-secret"
	spec := &v1.Volume{
		Name: "vol1-pvname",
		VolumeSource: v1.VolumeSource{
			StorageOS: &v1.StorageOSVolumeSource{
				VolumeName:      "vol1",
				VolumeNamespace: "ns1",
				FSType:          "ext3",
				SecretRef: &v1.LocalObjectReference{
					Name: secretName,
				},
			},
		},
	}

	client := fake.NewSimpleClientset()

	client.CoreV1().Secrets("default").Create(context.TODO(), &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: "default",
		},
		Type: "kubernetes.io/storageos",
		Data: map[string][]byte{
			"apiUsername": []byte("storageos"),
			"apiPassword": []byte("storageos"),
			"apiAddr":     []byte("tcp://localhost:5705"),
		}}, metav1.CreateOptions{})

	plug.(*storageosPlugin).host = volumetest.NewFakeVolumeHost(t, tmpDir, client, nil)

	// Test Mounter
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid"), Namespace: "default"}}
	fakeManager := &fakePDManager{}

	apiCfg, err := parsePodSecret(pod, secretName, plug.(*storageosPlugin).host.GetKubeClient())
	if err != nil {
		t.Errorf("Couldn't get secret from %v/%v", pod.Namespace, secretName)
	}

	mounter, err := plug.(*storageosPlugin).newMounterInternal(volume.NewSpecFromVolume(spec), pod, apiCfg, fakeManager, mount.NewFakeMounter(nil), &testingexec.FakeExec{})
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	expectedPath := filepath.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~storageos/vol1-pvname.ns1.vol1")
	volPath := mounter.GetPath()
	if volPath != expectedPath {
		t.Errorf("Expected path: '%s' got: '%s'", expectedPath, volPath)
	}

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volPath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	if !fakeManager.attachDeviceCalled {
		t.Errorf("AttachDevice not called")
	}
	if !fakeManager.attachCalled {
		t.Errorf("Attach not called")
	}
	if !fakeManager.mountCalled {
		t.Errorf("Mount not called")
	}

	// Test Unmounter
	fakeManager = &fakePDManager{}
	unmounter, err := plug.(*storageosPlugin).newUnmounterInternal("vol1-pvname", types.UID("poduid"), fakeManager, mount.NewFakeMounter(nil), &testingexec.FakeExec{})
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	volPath = unmounter.GetPath()
	if volPath != expectedPath {
		t.Errorf("Expected path: '%s' got: '%s'", expectedPath, volPath)
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volPath)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}

	if !fakeManager.unmountCalled {
		t.Errorf("Unmount not called")
	}
	if !fakeManager.detachCalled {
		t.Errorf("Detach not called")
	}

	// Test Provisioner
	fakeManager = &fakePDManager{}
	mountOptions := []string{"sync", "noatime"}
	options := volume.VolumeOptions{
		PVC: volumetest.CreateTestPVC("100Mi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		// PVName: "test-volume-name",
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
		Parameters: map[string]string{
			"VolumeNamespace":      "test-volume-namespace",
			"adminSecretName":      secretName,
			"adminsecretnamespace": "default",
		},
		MountOptions: mountOptions,
	}
	provisioner, err := plug.(*storageosPlugin).newProvisionerInternal(options, fakeManager)
	if err != nil {
		t.Errorf("newProvisionerInternal() failed: %v", err)
	}

	persistentSpec, err := provisioner.Provision(nil, nil)
	if err != nil {
		t.Fatalf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.StorageOS.VolumeName != "test-storageos-name" {
		t.Errorf("Provision() returned unexpected volume Name: %s, expected test-storageos-name", persistentSpec.Spec.PersistentVolumeSource.StorageOS.VolumeName)
	}
	if persistentSpec.Spec.PersistentVolumeSource.StorageOS.VolumeNamespace != "test-storageos-namespace" {
		t.Errorf("Provision() returned unexpected volume Namespace: %s", persistentSpec.Spec.PersistentVolumeSource.StorageOS.VolumeNamespace)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 100*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}
	if persistentSpec.Spec.PersistentVolumeSource.StorageOS.FSType != "ext2" {
		t.Errorf("Provision() returned unexpected volume FSType: %s", persistentSpec.Spec.PersistentVolumeSource.StorageOS.FSType)
	}
	if len(persistentSpec.Spec.MountOptions) != 2 {
		t.Errorf("Provision() returned unexpected volume mount options: %v", persistentSpec.Spec.MountOptions)
	}
	if persistentSpec.Labels["fakepdmanager"] != "yes" {
		t.Errorf("Provision() returned unexpected labels: %v", persistentSpec.Labels)
	}
	if !fakeManager.createCalled {
		t.Errorf("Create not called")
	}

	// Test Deleter
	fakeManager = &fakePDManager{}
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*storageosPlugin).newDeleterInternal(volSpec, apiCfg, fakeManager)
	if err != nil {
		t.Errorf("newDeleterInternal() failed: %v", err)
	}

	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
	if !fakeManager.deleteCalled {
		t.Errorf("Delete not called")
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("storageos_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				StorageOS: &v1.StorageOSPersistentVolumeSource{VolumeName: "pvA", VolumeNamespace: "vnsA", ReadOnly: false},
			},
			ClaimRef: &v1.ObjectReference{
				Name: "claimA",
			},
		},
	}

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase: v1.ClaimBound,
		},
	}

	client := fake.NewSimpleClientset(pv, claim)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, client, nil))
	plug, _ := plugMgr.FindPluginByName(storageosPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "nsA", UID: types.UID("poduid")}}
	fakeManager := &fakePDManager{}
	apiCfg := GetAPIConfig()
	mounter, err := plug.(*storageosPlugin).newMounterInternal(spec, pod, apiCfg, fakeManager, mount.NewFakeMounter(nil), &testingexec.FakeExec{})
	if err != nil {
		t.Fatalf("error creating a new internal mounter:%v", err)
	}
	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}
