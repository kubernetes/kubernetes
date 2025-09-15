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

package nfs

import (
	"os"
	"path/filepath"
	"testing"

	"k8s.io/mount-utils"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("nfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Fatal("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/nfs" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}

	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{NFS: &v1.NFSVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{NFS: &v1.NFSVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("nfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteOnce) || !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadOnlyMany) || !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteMany) {
		t.Errorf("Expected three AccessModeTypes:  %s, %s, and %s", v1.ReadWriteOnce, v1.ReadOnlyMany, v1.ReadWriteMany)
	}
}

func TestRecycler(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("nfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins([]volume.VolumePlugin{&nfsPlugin{nil, volume.VolumeConfig{}}}, nil, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	spec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{NFS: &v1.NFSVolumeSource{Path: "/foo"}}}}}
	_, pluginErr := plugMgr.FindRecyclablePluginBySpec(spec)
	if pluginErr != nil {
		t.Errorf("Can't find the plugin by name")
	}
}

func doTestPlugin(t *testing.T, spec *volume.Spec, expectedDevice string) {
	tmpDir, err := utiltesting.MkTmpdir("nfs_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fake := mount.NewFakeMounter(nil)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.(*nfsPlugin).newMounterInternal(spec, pod, fake)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volumePath := mounter.GetPath()
	expectedPath := filepath.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~nfs/vol1")
	if volumePath != expectedPath {
		t.Errorf("Unexpected path, expected %q, got: %q", expectedPath, volumePath)
	}
	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if mounter.(*nfsMounter).readOnly {
		t.Errorf("The volume source should not be read-only and it is.")
	}
	mntDevs, err := fake.List()
	if err != nil {
		t.Errorf("fakeMounter.List() failed: %v", err)
	}
	if len(mntDevs) != 1 {
		t.Errorf("unexpected number of mounted devices. expected: %v, got %v", 1, len(mntDevs))
	} else {
		if mntDevs[0].Type != "nfs" {
			t.Errorf("unexpected type of mounted devices. expected: %v, got %v", "nfs", mntDevs[0].Type)
		}
		if mntDevs[0].Device != expectedDevice {
			t.Errorf("unexpected nfs device, expected %q, got: %q", expectedDevice, mntDevs[0].Device)
		}
	}
	log := fake.GetLog()
	if len(log) != 1 {
		t.Errorf("Mount was not called exactly one time. It was called %d times.", len(log))
	} else {
		if log[0].Action != mount.FakeActionMount {
			t.Errorf("Unexpected mounter action: %#v", log[0])
		}
	}
	fake.ResetLog()

	unmounter, err := plug.(*nfsPlugin).newUnmounterInternal("vol1", types.UID("poduid"), fake)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}
	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}
	log = fake.GetLog()
	if len(log) != 1 {
		t.Errorf("Unmount was not called exactly one time. It was called %d times.", len(log))
	} else {
		if log[0].Action != mount.FakeActionUnmount {
			t.Errorf("Unexpected unmounter action: %#v", log[0])
		}
	}

	fake.ResetLog()
}

func TestPluginVolume(t *testing.T) {
	vol := &v1.Volume{
		Name:         "vol1",
		VolumeSource: v1.VolumeSource{NFS: &v1.NFSVolumeSource{Server: "localhost", Path: "/somepath", ReadOnly: false}},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol), "localhost:/somepath")
}

func TestIPV6VolumeSource(t *testing.T) {
	vol := &v1.Volume{
		Name:         "vol1",
		VolumeSource: v1.VolumeSource{NFS: &v1.NFSVolumeSource{Server: "0:0:0:0:0:0:0:1", Path: "/somepath", ReadOnly: false}},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol), "[0:0:0:0:0:0:0:1]:/somepath")
}

func TestIPV4VolumeSource(t *testing.T) {
	vol := &v1.Volume{
		Name:         "vol1",
		VolumeSource: v1.VolumeSource{NFS: &v1.NFSVolumeSource{Server: "127.0.0.1", Path: "/somepath", ReadOnly: false}},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol), "127.0.0.1:/somepath")
}

func TestPluginPersistentVolume(t *testing.T) {
	vol := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vol1",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				NFS: &v1.NFSVolumeSource{Server: "localhost", Path: "/somepath", ReadOnly: false},
			},
		},
	}

	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false), "localhost:/somepath")
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("nfs_test")
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
				NFS: &v1.NFSVolumeSource{},
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
	plugMgr.InitPlugins(ProbeVolumePlugins(volume.VolumeConfig{}), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, client, nil))
	plug, _ := plugMgr.FindPluginByName(nfsPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod)
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}
