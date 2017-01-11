// +build linux

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

package host_path

import (
	"fmt"
	"os"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/host-path" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if len(plug.GetAccessModes()) != 1 || plug.GetAccessModes()[0] != v1.ReadWriteOnce {
		t.Errorf("Expected %s PersistentVolumeAccessMode", v1.ReadWriteOnce)
	}
}

func TestRecycler(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	pluginHost := volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil)
	plugMgr.InitPlugins([]volume.VolumePlugin{&hostPathPlugin{nil, volume.VolumeConfig{}}}, pluginHost)

	spec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/foo"}}}}}
	plug, err := plugMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	recycler, err := plug.NewRecycler("pv-name", spec, nil)
	if err != nil {
		t.Errorf("Failed to make a new Recycler: %v", err)
	}
	if recycler.GetPath() != spec.PersistentVolume.Spec.HostPath.Path {
		t.Errorf("Expected %s but got %s", spec.PersistentVolume.Spec.HostPath.Path, recycler.GetPath())
	}
}

func TestDeleter(t *testing.T) {
	// Deleter has a hard-coded regex for "/tmp".
	tempPath := fmt.Sprintf("/tmp/hostpath/%s", uuid.NewUUID())
	defer os.RemoveAll(tempPath)
	err := os.MkdirAll(tempPath, 0750)
	if err != nil {
		t.Fatalf("Failed to create tmp directory for deleter: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil))

	spec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{Path: tempPath}}}}}
	plug, err := plugMgr.FindDeletablePluginBySpec(spec)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	deleter, err := plug.NewDeleter(spec)
	if err != nil {
		t.Errorf("Failed to make a new Deleter: %v", err)
	}
	if deleter.GetPath() != tempPath {
		t.Errorf("Expected %s but got %s", tempPath, deleter.GetPath())
	}
	if err := deleter.Delete(); err != nil {
		t.Errorf("Mock Recycler expected to return nil but got %s", err)
	}
	if exists, _ := util.FileExists("foo"); exists {
		t.Errorf("Temp path expected to be deleted, but was found at %s", tempPath)
	}
}

func TestDeleterTempDir(t *testing.T) {
	tests := map[string]struct {
		expectedFailure bool
		path            string
	}{
		"just-tmp": {true, "/tmp"},
		"not-tmp":  {true, "/nottmp"},
		"good-tmp": {false, "/tmp/scratch"},
	}

	for name, test := range tests {
		plugMgr := volume.VolumePluginMgr{}
		plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil))
		spec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{Path: test.path}}}}}
		plug, _ := plugMgr.FindDeletablePluginBySpec(spec)
		deleter, _ := plug.NewDeleter(spec)
		err := deleter.Delete()
		if err == nil && test.expectedFailure {
			t.Errorf("Expected failure for test '%s' but got nil err", name)
		}
		if err != nil && !test.expectedFailure {
			t.Errorf("Unexpected failure for test '%s': %v", name, err)
		}
	}
}

func TestProvisioner(t *testing.T) {
	tempPath := fmt.Sprintf("/tmp/hostpath/%s", uuid.NewUUID())
	defer os.RemoveAll(tempPath)
	err := os.MkdirAll(tempPath, 0750)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{ProvisioningEnabled: true}),
		volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil))
	spec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{Path: tempPath}}}}}
	plug, err := plugMgr.FindCreatablePluginBySpec(spec)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	options := volume.VolumeOptions{
		PVC: volumetest.CreateTestPVC("1Gi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}
	creater, err := plug.NewProvisioner(options)
	if err != nil {
		t.Errorf("Failed to make a new Provisioner: %v", err)
	}
	pv, err := creater.Provision()
	if err != nil {
		t.Errorf("Unexpected error creating volume: %v", err)
	}
	if pv.Spec.HostPath.Path == "" {
		t.Errorf("Expected pv.Spec.HostPath.Path to not be empty: %#v", pv)
	}
	expectedCapacity := resource.NewQuantity(1*1024*1024*1024, resource.BinarySI)
	actualCapacity := pv.Spec.Capacity[v1.ResourceStorage]
	expectedAmt := expectedCapacity.Value()
	actualAmt := actualCapacity.Value()
	if expectedAmt != actualAmt {
		t.Errorf("Expected capacity %+v but got %+v", expectedAmt, actualAmt)
	}

	if pv.Spec.PersistentVolumeReclaimPolicy != v1.PersistentVolumeReclaimDelete {
		t.Errorf("Expected reclaim policy %+v but got %+v", v1.PersistentVolumeReclaimDelete, pv.Spec.PersistentVolumeReclaimPolicy)
	}

	os.RemoveAll(pv.Spec.HostPath.Path)
}

func TestPlugin(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name:         "vol1",
		VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/vol1"}},
	}
	pod := &v1.Pod{ObjectMeta: v1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	path := mounter.GetPath()
	if path != "/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	unmounter, err := plug.NewUnmounter("vol1", types.UID("poduid"))
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &v1.PersistentVolume{
		ObjectMeta: v1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "foo"},
			},
			ClaimRef: &v1.ObjectReference{
				Name: "claimA",
			},
		},
	}

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: v1.ObjectMeta{
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
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("/tmp/fake", client, nil))
	plug, _ := plugMgr.FindPluginByName(hostPathPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: v1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}
