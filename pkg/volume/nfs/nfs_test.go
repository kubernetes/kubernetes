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

package nfs

import (
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("fake", nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/nfs" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{NFS: &api.NFSVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{NFS: &api.NFSVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plug.GetAccessModes(), api.ReadWriteOnce) || !contains(plug.GetAccessModes(), api.ReadOnlyMany) || !contains(plug.GetAccessModes(), api.ReadWriteMany) {
		t.Errorf("Expected three AccessModeTypes:  %s, %s, and %s", api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany)
	}
}

func TestRecycler(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins([]volume.VolumePlugin{&nfsPlugin{nil, newMockRecycler}}, volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	spec := &volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{NFS: &api.NFSVolumeSource{Path: "/foo"}}}}}
	plug, err := plugMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	recycler, err := plug.NewRecycler(spec)
	if err != nil {
		t.Errorf("Failed to make a new Recyler: %v", err)
	}
	if recycler.GetPath() != spec.PersistentVolume.Spec.NFS.Path {
		t.Errorf("Expected %s but got %s", spec.PersistentVolume.Spec.NFS.Path, recycler.GetPath())
	}
	if err := recycler.Recycle(); err != nil {
		t.Errorf("Mock Recycler expected to return nil but got %s", err)
	}
}

func newMockRecycler(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error) {
	return &mockRecycler{
		path: spec.PersistentVolume.Spec.NFS.Path,
	}, nil
}

type mockRecycler struct {
	path string
	host volume.VolumeHost
}

func (r *mockRecycler) GetPath() string {
	return r.path
}

func (r *mockRecycler) Recycle() error {
	// return nil means recycle passed
	return nil
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fake := &mount.FakeMounter{}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.(*nfsPlugin).newBuilderInternal(spec, pod, fake)
	volumePath := builder.GetPath()
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}
	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~nfs/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}
	if err := builder.SetUp(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if builder.(*nfsBuilder).readOnly {
		t.Errorf("The volume source should not be read-only and it is.")
	}
	if len(fake.Log) != 1 {
		t.Errorf("Mount was not called exactly one time. It was called %d times.", len(fake.Log))
	} else {
		if fake.Log[0].Action != mount.FakeActionMount {
			t.Errorf("Unexpected mounter action: %#v", fake.Log[0])
		}
	}
	fake.ResetLog()

	cleaner, err := plug.(*nfsPlugin).newCleanerInternal("vol1", types.UID("poduid"), fake)
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}
	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if len(fake.Log) != 1 {
		t.Errorf("Unmount was not called exactly one time. It was called %d times.", len(fake.Log))
	} else {
		if fake.Log[0].Action != mount.FakeActionUnmount {
			t.Errorf("Unexpected mounter action: %#v", fake.Log[0])
		}
	}

	fake.ResetLog()
}

func TestPluginVolume(t *testing.T) {
	vol := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{NFS: &api.NFSVolumeSource{Server: "localhost", Path: "/tmp", ReadOnly: false}},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol))
}

func TestPluginPersistentVolume(t *testing.T) {
	vol := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "vol1",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				NFS: &api.NFSVolumeSource{Server: "localhost", Path: "/tmp", ReadOnly: false},
			},
		},
	}

	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				NFS: &api.NFSVolumeSource{},
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
	client.AddReactor("*", "*", testclient.ObjectReaction(o, latest.RESTMapper))

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", client, nil))
	plug, _ := plugMgr.FindPluginByName(nfsPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its builder creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, _ := plug.NewBuilder(spec, pod, volume.VolumeOptions{}, nil)

	if !builder.IsReadOnly() {
		t.Errorf("Expected true for builder.IsReadOnly")
	}
}
