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

package glusterfs

import (
	"fmt"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("glusterfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/glusterfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/glusterfs" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{}}}}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("glusterfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/glusterfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plug.GetAccessModes(), api.ReadWriteOnce) || !contains(plug.GetAccessModes(), api.ReadOnlyMany) || !contains(plug.GetAccessModes(), api.ReadWriteMany) {
		t.Errorf("Expected three AccessModeTypes:  %s, %s, and %s", api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany)
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

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	tmpDir, err := utiltesting.MkTmpdir("glusterfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/glusterfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	ep := &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "foo"}, Subsets: []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}}}}}
	var fcmd exec.FakeCmd
	fcmd = exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// mount
			func() ([]byte, error) {
				return []byte{}, nil
			},
		},
	}
	fake := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.(*glusterfsPlugin).newMounterInternal(spec, ep, pod, &mount.FakeMounter{}, &fake)
	volumePath := mounter.GetPath()
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Error("Got a nil Mounter")
	}
	path := mounter.GetPath()
	expectedPath := fmt.Sprintf("%s/pods/poduid/volumes/kubernetes.io~glusterfs/vol1", tmpDir)
	if path != expectedPath {
		t.Errorf("Unexpected path, expected %q, got: %q", expectedPath, path)
	}
	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	unmounter, err := plug.(*glusterfsPlugin).newUnmounterInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Error("Got a nil Unmounter")
	}
	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}

func TestPluginVolume(t *testing.T) {
	vol := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{EndpointsName: "ep", Path: "vol", ReadOnly: false}},
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
				Glusterfs: &api.GlusterfsVolumeSource{EndpointsName: "ep", Path: "vol", ReadOnly: false},
			},
		},
	}

	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("glusterfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				Glusterfs: &api.GlusterfsVolumeSource{EndpointsName: "ep", Path: "vol", ReadOnly: false},
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

	ep := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Namespace: "nsA",
			Name:      "ep",
		},
		Subsets: []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}},
			Ports:     []api.EndpointPort{{Name: "foo", Port: 80, Protocol: api.ProtocolTCP}},
		}},
	}

	client := fake.NewSimpleClientset(pv, claim, ep)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, client, nil, "" /* rootContext */))
	plug, _ := plugMgr.FindPluginByName(glusterfsPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "nsA", UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}
