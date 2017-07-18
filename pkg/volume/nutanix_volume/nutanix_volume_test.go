/*
Copyright 2014 The Kubernetes Authors.

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

package nutanix_volume

import (
	"fmt"
	"os"
	"path"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("ntnxvolTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/nutanix-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/nutanix-volume" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: 
		&v1.Volume{VolumeSource: v1.VolumeSource{NutanixVolume:
			&v1.NutanixVolumeSource{
				User: "testUser",
				Password: "testPwd",
			}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume:
		&v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource:
			v1.PersistentVolumeSource{NutanixVolume: 
				&v1.NutanixVolumeSource{
					User: "testUser",
					Password: "testPwd",
				}}}}}) {
					t.Errorf("Expected true")
				}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("ntnxvolTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/nutanix-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plug.GetAccessModes(), v1.ReadWriteOnce) || !contains(plug.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", v1.ReadWriteOnce, v1.ReadOnlyMany)
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

type fakeDiskManager struct {
	tmpDir       string
}

func (fake *fakeDiskManager) MakeGlobalPDName(disk nutanixVolume, target string) string {
	return fake.tmpDir
}

func (fake *fakeDiskManager) AttachDisk(b nutanixVolumeMounter) error {
	return nil
}

func (fake *fakeDiskManager) DetachDisk(c nutanixVolumeUnmounter, mntPath string) error {
	return nil
}

func CreateVolume() (r *v1.NutanixVolumeSource, size int64, err error) {
	return &v1.NutanixVolumeSource{
		VolumeName: "test-ntnx-volume-name",
		FSType: "ext4",
		ReadOnly: false,
		User: "testUser",
		Password: "testPwd",
	}, 100*1024*1024*1024, nil
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("ntnxvolTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/nutanix-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			NutanixVolume: &v1.NutanixVolumeSource{
				VolumeName: "vol1",
				FSType: "ext4",
				ReadOnly: false,
				User: "testUser",
				Password: "testPwd",
			},
		},
	}

	// Test mount
	fakeManager := &fakeDiskManager{}
	fakeMounter := &mount.FakeMounter{}
	mounter, err := plug.(*nutanixVolumePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~nutanix-volume/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	// Test Unmount
	unmounter, err := plug.(*nutanixVolumePlugin).newUnmounterInternal("vol1", types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	// Test Provisioner
	pvc := volumetest.CreateTestPVC("100Mi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce})
	scName := "silver"
	pvc.Spec.StorageClassName = &scName
	options := volume.VolumeOptions{
		PVC: pvc,
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
		Parameters: map[string]string{
			"User": "testUser",
			"Password": "testPwd",
			"DataServiceEndPoint": "127.0.0.1:3260",
			"PrismEndPoint": "127.0.0.2:9440",
			"StorageContainer": "default-container",
			"FSType": "xfs",
		},
	}
	_, err = plug.(*nutanixVolumePlugin).newProvisionerInternal(options)
	if err != nil {
		t.Errorf("newProvisionerInternal failed: %v", err)
	}
	
	ntnx_provisioner := &nutanixVolumeProvisioner{
		plugin: plug.(*nutanixVolumePlugin),
		options: options,
	}

	_, err = parseClassParameters(ntnx_provisioner.options.Parameters, ntnx_provisioner.plugin.host.GetKubeClient())
	if err != nil {
		t.Errorf("parseClassParameters failed: %v", err)
	}

	volumeSource, size, err := CreateVolume()
	persistentSpec := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-ntnx-volume-name",
			Annotations: map[string]string{
				"kubernetes.io/createdby": "nutanix-volume-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: ntnx_provisioner.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   ntnx_provisioner.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dKi", size/1024)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				NutanixVolume: volumeSource,
			},
		},
	}
	
	if persistentSpec.Spec.PersistentVolumeSource.NutanixVolume.VolumeName != "test-ntnx-volume-name" {
		t.Errorf("Provision() returned unexpected volume Name: %s",
			 persistentSpec.Spec.PersistentVolumeSource.NutanixVolume.VolumeName)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size1 := cap.Value()
	if size1 != 100*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	_, err = plug.(*nutanixVolumePlugin).newDeleterInternal(volSpec)
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				NutanixVolume: &v1.NutanixVolumeSource{},
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

	tmpDir, err := utiltesting.MkTmpdir("ntnxvolTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, client, nil))
	plug, _ := plugMgr.FindPluginByName(nutanixVolumePluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}
