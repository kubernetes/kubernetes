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

package aws_ebs

import (
	"fmt"
	"os"
	"path"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("awsebsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/aws-ebs" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("awsebsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	if !contains(plug.GetAccessModes(), api.ReadWriteOnce) {
		t.Errorf("Expected to support AccessModeTypes:  %s", api.ReadWriteOnce)
	}
	if contains(plug.GetAccessModes(), api.ReadOnlyMany) {
		t.Errorf("Expected not to support AccessModeTypes:  %s", api.ReadOnlyMany)
	}
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

type fakePDManager struct {
	attachCalled bool
	detachCalled bool
}

// TODO(jonesdl) To fully test this, we could create a loopback device
// and mount that instead.
func (fake *fakePDManager) AttachAndMountDisk(b *awsElasticBlockStoreBuilder, globalPDPath string) error {
	globalPath := makeGlobalPDPath(b.plugin.host, b.volumeID)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return err
	}
	fake.attachCalled = true
	// Simulate the global mount so that the fakeMounter returns the
	// expected number of mounts for the attached disk.
	b.mounter.Mount(globalPath, globalPath, b.fsType, nil)
	return nil
}

func (fake *fakePDManager) DetachDisk(c *awsElasticBlockStoreCleaner) error {
	globalPath := makeGlobalPDPath(c.plugin.host, c.volumeID)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	fake.detachCalled = true
	return nil
}

func (fake *fakePDManager) CreateVolume(c *awsElasticBlockStoreProvisioner) (volumeID string, volumeSizeGB int, labels map[string]string, err error) {
	labels = make(map[string]string)
	labels["fakepdmanager"] = "yes"
	return "test-aws-volume-name", 100, labels, nil
}

func (fake *fakePDManager) DeleteVolume(cd *awsElasticBlockStoreDeleter) error {
	if cd.volumeID != "test-aws-volume-name" {
		return fmt.Errorf("Deleter got unexpected volume name: %s", cd.volumeID)
	}
	return nil
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("awsebsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
				VolumeID: "pd",
				FSType:   "ext4",
			},
		},
	}
	fakeManager := &fakePDManager{}
	fakeMounter := &mount.FakeMounter{}
	builder, err := plug.(*awsElasticBlockStorePlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~aws-ebs/vol1")
	path := builder.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := builder.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if !fakeManager.attachCalled {
		t.Errorf("Attach watch not called")
	}

	fakeManager = &fakePDManager{}
	cleaner, err := plug.(*awsElasticBlockStorePlugin).newCleanerInternal("vol1", types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if !fakeManager.detachCalled {
		t.Errorf("Detach watch not called")
	}

	// Test Provisioner
	cap := resource.MustParse("100Mi")
	options := volume.VolumeOptions{
		Capacity: cap,
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
		},
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}
	provisioner, err := plug.(*awsElasticBlockStorePlugin).newProvisionerInternal(options, &fakePDManager{})
	persistentSpec, err := provisioner.NewPersistentVolumeTemplate()
	if err != nil {
		t.Errorf("NewPersistentVolumeTemplate() failed: %v", err)
	}

	// get 2nd Provisioner - persistent volume controller will do the same
	provisioner, err = plug.(*awsElasticBlockStorePlugin).newProvisionerInternal(options, &fakePDManager{})
	err = provisioner.Provision(persistentSpec)
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.AWSElasticBlockStore.VolumeID != "test-aws-volume-name" {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.AWSElasticBlockStore.VolumeID)
	}
	cap = persistentSpec.Spec.Capacity[api.ResourceStorage]
	size := cap.Value()
	if size != 100*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	if persistentSpec.Labels["fakepdmanager"] != "yes" {
		t.Errorf("Provision() returned unexpected labels: %v", persistentSpec.Labels)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*awsElasticBlockStorePlugin).newDeleterInternal(volSpec, &fakePDManager{})
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{},
			},
			ClaimRef: &api.ObjectReference{
				Name: "claimA",
			},
		},
	}

	claim := &api.PersistentVolumeClaim{
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
	}

	clientset := fake.NewSimpleClientset(pv, claim)

	tmpDir, err := utiltesting.MkTmpdir("awsebsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, clientset, nil))
	plug, _ := plugMgr.FindPluginByName(awsElasticBlockStorePluginName)

	// readOnly bool is supplied by persistent-claim volume source when its builder creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, _ := plug.NewBuilder(spec, pod, volume.VolumeOptions{})

	if !builder.GetAttributes().ReadOnly {
		t.Errorf("Expected true for builder.IsReadOnly")
	}
}

func TestBuilderAndCleanerTypeAssert(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("awsebsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
				VolumeID: "pd",
				FSType:   "ext4",
			},
		},
	}

	builder, err := plug.(*awsElasticBlockStorePlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), &fakePDManager{}, &mount.FakeMounter{})
	if _, ok := builder.(volume.Cleaner); ok {
		t.Errorf("Volume Builder can be type-assert to Cleaner")
	}

	cleaner, err := plug.(*awsElasticBlockStorePlugin).newCleanerInternal("vol1", types.UID("poduid"), &fakePDManager{}, &mount.FakeMounter{})
	if _, ok := cleaner.(volume.Builder); ok {
		t.Errorf("Volume Cleaner can be type-assert to Builder")
	}
}
