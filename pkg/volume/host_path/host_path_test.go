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

package host_path

import (
	"fmt"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/host-path" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if len(plug.GetAccessModes()) != 1 || plug.GetAccessModes()[0] != api.ReadWriteOnce {
		t.Errorf("Expected %s PersistentVolumeAccessMode", api.ReadWriteOnce)
	}
}

func TestRecycler(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	pluginHost := volume.NewFakeVolumeHost("/tmp/fake", nil, nil)
	plugMgr.InitPlugins([]volume.VolumePlugin{&hostPathPlugin{nil, volume.NewFakeRecycler, nil, nil, volume.VolumeConfig{}}}, pluginHost)

	spec := &volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/foo"}}}}}
	plug, err := plugMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	recycler, err := plug.NewRecycler(spec)
	if err != nil {
		t.Errorf("Failed to make a new Recyler: %v", err)
	}
	if recycler.GetPath() != spec.PersistentVolume.Spec.HostPath.Path {
		t.Errorf("Expected %s but got %s", spec.PersistentVolume.Spec.HostPath.Path, recycler.GetPath())
	}
	if err := recycler.Recycle(); err != nil {
		t.Errorf("Mock Recycler expected to return nil but got %s", err)
	}
}

func TestDeleter(t *testing.T) {
	tempPath := fmt.Sprintf("/tmp/hostpath/%s", util.NewUUID())
	defer os.RemoveAll(tempPath)
	err := os.MkdirAll(tempPath, 0750)
	if err != nil {
		t.Fatal("Failed to create tmp directory for deleter: %v", err)
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	spec := &volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{Path: tempPath}}}}}
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
		plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))
		spec := &volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{Path: test.path}}}}}
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

func TestCreater(t *testing.T) {
	tempPath := "/tmp/hostpath/"
	defer os.RemoveAll(tempPath)
	err := os.MkdirAll(tempPath, 0750)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))
	spec := &volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{Path: tempPath}}}}}
	plug, err := plugMgr.FindCreatablePluginBySpec(spec)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	creater, err := plug.NewCreater(volume.VolumeOptions{CapacityMB: 100, PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete})
	if err != nil {
		t.Errorf("Failed to make a new Creater: %v", err)
	}
	pv, err := creater.Create()
	if err != nil {
		t.Errorf("Unexpected error creating volume: %v", err)
	}
	if pv.Spec.HostPath.Path == "" {
		t.Errorf("Expected pv.Spec.HostPath.Path to not be empty: %#v", pv)
	}
	expectedCapacity := resource.NewQuantity(100*1024*1024, resource.BinarySI)
	actualCapacity := pv.Spec.Capacity[api.ResourceStorage]
	expectedAmt := expectedCapacity.Value()
	actualAmt := actualCapacity.Value()
	if expectedAmt != actualAmt {
		t.Errorf("Expected capacity %+v but got %+v", expectedAmt, actualAmt)
	}

	if pv.Spec.PersistentVolumeReclaimPolicy != api.PersistentVolumeReclaimDelete {
		t.Errorf("Expected reclaim policy %+v but got %+v", api.PersistentVolumeReclaimDelete, pv.Spec.PersistentVolumeReclaimPolicy)
	}

	os.RemoveAll(pv.Spec.HostPath.Path)
}

func TestPlugin(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/vol1"}},
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.NewBuilder(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	path := builder.GetPath()
	if path != "/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := builder.SetUp(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	cleaner, err := plug.NewCleaner("vol1", types.UID("poduid"))
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "foo"},
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

	o := testclient.NewObjects(api.Scheme, api.Scheme)
	o.Add(pv)
	o.Add(claim)
	client := &testclient.Fake{}
	client.AddReactor("*", "*", testclient.ObjectReaction(o, testapi.Default.RESTMapper()))

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", client, nil))
	plug, _ := plugMgr.FindPluginByName(hostPathPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its builder creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, _ := plug.NewBuilder(spec, pod, volume.VolumeOptions{})

	if !builder.IsReadOnly() {
		t.Errorf("Expected true for builder.IsReadOnly")
	}
}
