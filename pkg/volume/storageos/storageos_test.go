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
	"fmt"
	"os"
	"path"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/util/mount"
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

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
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{StorageOS: &v1.StorageOSVolumeSource{}}}}}) {
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/storageos")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plug.GetAccessModes(), v1.ReadWriteOnce) || !contains(plug.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", v1.ReadWriteOnce, v1.ReadOnlyMany)
	}
}

type fakePDManager struct {
	attachCalled  bool
	detachCalled  bool
	mountCalled   bool
	unmountCalled bool
	createCalled  bool
	deleteCalled  bool
}

func (fake *fakePDManager) CreateVolume(p *storageosProvisioner) (*storageosVolume, error) {
	fake.createCalled = true
	labels := make(map[string]string)
	labels["fakepdmanager"] = "yes"
	return &storageosVolume{
		Name:      "test-storageos-name",
		Namespace: "test-storageos-namespace",
		Pool:      "test-storageos-pool",
		Size:      100,
		Labels:    labels,
		FSType:    "ext2",
	}, nil
}

func (fake *fakePDManager) AttachVolume(b *storageosMounter) (string, error) {
	fake.attachCalled = true
	return "", nil
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
	// return util.UnmountPath(mountPath, b.mounter)
	return nil
}

func (fake *fakePDManager) DeleteVolume(d *storageosDeleter) error {
	fake.deleteCalled = true
	if d.volName != "test-storageos-name" {
		return fmt.Errorf("Deleter got unexpected volume name: %s", d.volName)
	}
	return nil
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("storageos_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/storageos")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1-pvname",
		VolumeSource: v1.VolumeSource{
			StorageOS: &v1.StorageOSVolumeSource{
				VolumeName: "vol1",
				Namespace:  "ns1",
				FSType:     "ext3",
			},
		},
	}

	// Test Mounter
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	fakeManager := &fakePDManager{}
	mounter, err := plug.(*storageosPlugin).newMounterInternal(volume.NewSpecFromVolume(spec), pod, fakeManager, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	expectedPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~storageos/ns1~vol1")
	volPath := mounter.GetPath()
	if volPath != expectedPath {
		t.Errorf("Got unexpected path: %s", volPath)
	}

	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volPath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	if !fakeManager.attachCalled {
		t.Errorf("Attach not called")
	}
	if !fakeManager.mountCalled {
		t.Errorf("Mount not called")
	}

	// Test Unmounter
	fakeManager = &fakePDManager{}
	unmounter, err := plug.(*storageosPlugin).newUnmounterInternal("ns1~vol1", types.UID("poduid"), fakeManager, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	volPath = unmounter.GetPath()
	if volPath != expectedPath {
		t.Errorf("Got unexpected path: %s", volPath)
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volPath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}

	if !fakeManager.unmountCalled {
		t.Errorf("Unmount not called")
	}
	if !fakeManager.detachCalled {
		t.Errorf("Detach not called")
	}

	// Test Provisioner
	fakeManager = &fakePDManager{}
	options := volume.VolumeOptions{
		PVC: volumetest.CreateTestPVC("100Mi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}
	provisioner, err := plug.(*storageosPlugin).newProvisionerInternal(options, fakeManager)
	if err != nil {
		t.Errorf("newProvisionerInternal() failed: %v", err)
	}
	persistentSpec, err := provisioner.Provision()
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.StorageOS.VolumeName != "test-storageos-name" {
		t.Errorf("Provision() returned unexpected volume Name: %s", persistentSpec.Spec.PersistentVolumeSource.StorageOS.VolumeName)
	}
	if persistentSpec.Spec.PersistentVolumeSource.StorageOS.Namespace != "test-storageos-namespace" {
		t.Errorf("Provision() returned unexpected volume Namespace: %s", persistentSpec.Spec.PersistentVolumeSource.StorageOS.Namespace)
	}
	if persistentSpec.Spec.PersistentVolumeSource.StorageOS.Pool != "test-storageos-pool" {
		t.Errorf("Provision() returned unexpected volume Pool: %s", persistentSpec.Spec.PersistentVolumeSource.StorageOS.Pool)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 100*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}
	if persistentSpec.Spec.PersistentVolumeSource.StorageOS.FSType != "ext2" {
		t.Errorf("Provision() returned unexpected volume FSType: %s", persistentSpec.Spec.PersistentVolumeSource.StorageOS.FSType)
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
	deleter, err := plug.(*storageosPlugin).newDeleterInternal(volSpec, fakeManager)
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

func contains(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
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
				StorageOS: &v1.StorageOSVolumeSource{VolumeName: "pvA", Namespace: "vnsA", ReadOnly: false},
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, client, nil))
	plug, _ := plugMgr.FindPluginByName(storageosPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "nsA", UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}
