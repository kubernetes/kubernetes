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

package awsebs

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/util/mount"
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/aws-ebs" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{}}}}}) {
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	if !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteOnce) {
		t.Errorf("Expected to support AccessModeTypes:  %s", v1.ReadWriteOnce)
	}
	if volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected not to support AccessModeTypes:  %s", v1.ReadOnlyMany)
	}
}

type fakePDManager struct {
}

// TODO(jonesdl) To fully test this, we could create a loopback device
// and mount that instead.
func (fake *fakePDManager) CreateVolume(c *awsElasticBlockStoreProvisioner, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (volumeID aws.KubernetesVolumeID, volumeSizeGB int, labels map[string]string, fstype string, err error) {
	labels = make(map[string]string)
	labels["fakepdmanager"] = "yes"
	return "test-aws-volume-name", 100, labels, "", nil
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
				VolumeID: "pd",
				FSType:   "ext4",
			},
		},
	}
	fakeManager := &fakePDManager{}
	fakeMounter := &mount.FakeMounter{}
	mounter, err := plug.(*awsElasticBlockStorePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volPath := filepath.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~aws-ebs/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	fakeManager = &fakePDManager{}
	unmounter, err := plug.(*awsElasticBlockStorePlugin).newUnmounterInternal("vol1", types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}

	// Test Provisioner
	options := volume.VolumeOptions{
		PVC:                           volumetest.CreateTestPVC("100Mi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}
	provisioner, err := plug.(*awsElasticBlockStorePlugin).newProvisionerInternal(options, &fakePDManager{})
	if err != nil {
		t.Errorf("Error creating new provisioner:%v", err)
	}

	persistentSpec, err := provisioner.Provision(nil, nil)
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.AWSElasticBlockStore.VolumeID != "test-aws-volume-name" {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.AWSElasticBlockStore.VolumeID)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 100*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	if persistentSpec.Labels["fakepdmanager"] != "yes" {
		t.Errorf("Provision() returned unexpected labels: %v", persistentSpec.Labels)
	}

	// check nodeaffinity members
	if persistentSpec.Spec.NodeAffinity == nil {
		t.Errorf("Provision() returned unexpected nil NodeAffinity")
	}

	if persistentSpec.Spec.NodeAffinity.Required == nil {
		t.Errorf("Provision() returned unexpected nil NodeAffinity.Required")
	}

	n := len(persistentSpec.Spec.NodeAffinity.Required.NodeSelectorTerms)
	if n != 1 {
		t.Errorf("Provision() returned unexpected number of NodeSelectorTerms %d. Expected %d", n, 1)
	}

	n = len(persistentSpec.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions)
	if n != 1 {
		t.Errorf("Provision() returned unexpected number of MatchExpressions %d. Expected %d", n, 1)
	}

	req := persistentSpec.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions[0]
	if req.Key != "fakepdmanager" {
		t.Errorf("Provision() returned unexpected requirement key in NodeAffinity %v", req.Key)
	}

	if req.Operator != v1.NodeSelectorOpIn {
		t.Errorf("Provision() returned unexpected requirement operator in NodeAffinity %v", req.Operator)
	}

	if len(req.Values) != 1 || req.Values[0] != "yes" {
		t.Errorf("Provision() returned unexpected requirement value in NodeAffinity %v", req.Values)
	}
	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*awsElasticBlockStorePlugin).newDeleterInternal(volSpec, &fakePDManager{})
	if err != nil {
		t.Errorf("Error creating new deleter:%v", err)
	}
	err = deleter.Delete()
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
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{},
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

	clientset := fake.NewSimpleClientset(pv, claim)

	tmpDir, err := utiltesting.MkTmpdir("awsebsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, clientset, nil))
	plug, _ := plugMgr.FindPluginByName(awsElasticBlockStorePluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}

func TestMounterAndUnmounterTypeAssert(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("awsebsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
				VolumeID: "pd",
				FSType:   "ext4",
			},
		},
	}

	mounter, err := plug.(*awsElasticBlockStorePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), &fakePDManager{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Error creating new mounter:%v", err)
	}
	if _, ok := mounter.(volume.Unmounter); ok {
		t.Errorf("Volume Mounter can be type-assert to Unmounter")
	}

	unmounter, err := plug.(*awsElasticBlockStorePlugin).newUnmounterInternal("vol1", types.UID("poduid"), &fakePDManager{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Error creating new unmounter:%v", err)
	}
	if _, ok := unmounter.(volume.Mounter); ok {
		t.Errorf("Volume Unmounter can be type-assert to Mounter")
	}
}

func TestMountOptions(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("aws-ebs")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/aws-ebs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{},
			},
			ClaimRef: &v1.ObjectReference{
				Name: "claimA",
			},
			MountOptions: []string{"_netdev"},
		},
	}

	fakeManager := &fakePDManager{}
	fakeMounter := &mount.FakeMounter{}

	mounter, err := plug.(*awsElasticBlockStorePlugin).newMounterInternal(volume.NewSpecFromPersistentVolume(pv, false), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	mountOptions := fakeMounter.MountPoints[0].Opts
	expectedMountOptions := []string{"_netdev", "bind"}
	if !reflect.DeepEqual(mountOptions, expectedMountOptions) {
		t.Errorf("Expected mount options to be %v got %v", expectedMountOptions, mountOptions)
	}
}
