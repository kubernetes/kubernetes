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

package digitalocean

import (
	"fmt"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/digitalocean/godo"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("dovolume_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName(doVolumePluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != doVolumePluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{DOVolume: &v1.DOVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{DOVolume: &v1.DOVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("dovolume_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName(doVolumePluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	if !contains(plug.GetAccessModes(), v1.ReadWriteOnce) {
		t.Errorf("Expected to support AccessModeTypes:  %s", v1.ReadWriteOnce)
	}
	if contains(plug.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected not to support AccessModeTypes:  %s", v1.ReadOnlyMany)
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

type fakeVolManager struct {
}

func (fake *fakeVolManager) DeleteVolume(volumeID string) error {
	if strings.HasPrefix(volumeID, "fake-volume-id-") {
		return nil
	}
	return fmt.Errorf("Deleter got unexpected volume id: %s", volumeID)
}

func (fake *fakeVolManager) CreateVolume(name, description string, sizeGB int) (string, error) {
	return fmt.Sprintf("fake-volume-id-%s", name), nil
}

func (fake *fakeVolManager) AttachVolume(volumeID string, dropletID int) (string, error) {
	if !strings.HasPrefix(volumeID, "fake-volume-id-") {
		return "", fmt.Errorf("Attacher got unexpected volume id: %s", volumeID)
	}
	if dropletID == 0 {
		return "", fmt.Errorf("Attacher got unexpected droplet id 0")
	}

	return "/dev/disk/by-id/scsi-0DO_Volume_" + volumeID, nil
}

func (fake *fakeVolManager) FindDropletForNode(node *v1.Node) (*godo.Droplet, error) {
	if !strings.HasPrefix(node.Name, "fake-node-name-") {
		return nil, fmt.Errorf("Find Droplet for Node got unexpected node name: %s", node.Name)
	}

	return &godo.Droplet{ID: 42}, nil
}

func (fake *fakeVolManager) GetDroplet(dropletID int) (*godo.Droplet, error) {
	if dropletID == 0 {
		return nil, fmt.Errorf("GetDroplet got unexpected droplet id 0")
	}

	return &godo.Droplet{ID: dropletID}, nil
}

func (fake *fakeVolManager) DisksAreAttached(volumeIDs []string, dropletID int) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, volumeID := range volumeIDs {
		attached[volumeID] = false
	}
	_, err := fake.GetDroplet(dropletID)
	if err != nil {
		return attached, err
	}

	for _, volumeID := range volumeIDs {
		if !strings.HasPrefix(volumeID, "fake-volume-id-") {
			attached[volumeID] = true
		}
	}

	return attached, nil
}

func (fake *fakeVolManager) DetachVolume(volumeID string, dropletID int) error {
	if !strings.HasPrefix(volumeID, "fake-volume-id-") {
		return fmt.Errorf("Detacher got unexpected volume id: %s", volumeID)
	}
	if dropletID == 0 {
		return fmt.Errorf("Detacher got unexpected droplet id 0")
	}

	return nil
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	tmpDir, err := utiltesting.MkTmpdir("dovolume_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName(doVolumePluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	fakeMounter := &mount.FakeMounter{}
	mounter, err := plug.(*doVolumePlugin).newMounterInternal(spec, types.UID("poduid"), fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	expectedPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~digitalocean-volume/vol1")
	path := mounter.GetPath()
	if path != expectedPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	err = mounter.SetUp(nil)
	if err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	_, err = os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	_, err = os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	unmounter := plug.(*doVolumePlugin).newUnmounterInternal("vol1", types.UID("poduid"), fakeMounter)
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter: %v", err)
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}

	// Test Provisioner
	options := volume.VolumeOptions{
		PVC: volumetest.CreateTestPVC("50Gi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}

	provisioner := plug.(*doVolumePlugin).newProvisionerInternal(options, &fakeVolManager{})
	persistentSpec, err := provisioner.Provision()
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if !strings.HasPrefix(
		persistentSpec.Spec.PersistentVolumeSource.DOVolume.VolumeID,
		"fake-volume-id-") {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.DOVolume.VolumeID)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 50*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*doVolumePlugin).newDeleterInternal(volSpec, &fakeVolManager{})
	if err != nil {
		t.Errorf("Deleter creation failed deleter: %v", err)
	}
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}

}

func TestPluginVolume(t *testing.T) {

	vol := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			DOVolume: &v1.DOVolumeSource{
				VolumeID: "fake-volume-id-vol1",
				FSType:   "ext4",
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol))
}

func TestPluginPersistentVolume(t *testing.T) {

	vol := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vol1",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				DOVolume: &v1.DOVolumeSource{
					VolumeID: "fake-volume-id-vol1",
					FSType:   "ext4",
				},
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestMounterAndUnmounterTypeAssert(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("dovolume_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName(doVolumePluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			DOVolume: &v1.DOVolumeSource{
				VolumeID: "fake-volume-id-test",
				FSType:   "ext4",
			},
		},
	}

	mounter, err := plug.(*doVolumePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), &mount.FakeMounter{})
	if _, ok := mounter.(volume.Unmounter); ok {
		t.Errorf("Volume Mounter can be type-assert to Unmounter")
	}

	unmounter := plug.(*doVolumePlugin).newUnmounterInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
	if _, ok := unmounter.(volume.Mounter); ok {
		t.Errorf("Volume Unmounter can be type-assert to Mounter")
	}
}
