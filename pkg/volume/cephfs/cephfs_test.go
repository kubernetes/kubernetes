/*
Copyright 2015 The Kubernetes Authors.

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

package cephfs

import (
	"os"
	"path"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cephTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/cephfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/cephfs" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{CephFS: &v1.CephFSVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cephTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/cephfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			CephFS: &v1.CephFSVolumeSource{
				Monitors:   []string{"a", "b"},
				User:       "user",
				SecretRef:  nil,
				SecretFile: "/etc/ceph/user.secret",
			},
		},
	}

	mounter, err := plug.(*cephfsPlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), &mount.FakeMounter{}, "secrets")
	volumePath := mounter.GetPath()
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volpath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~cephfs/vol1")
	path := mounter.GetPath()
	if path != volpath {
		t.Errorf("Got unexpected path: %s", path)
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
	unmounter, err := plug.(*cephfsPlugin).newUnmounterInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
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
		t.Errorf("SetUp() failed: %v", err)
	}
}

func TestConstructVolumeSpec(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cephTest")
	if err != nil {
		t.Fatalf("Can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/cephfs")
	if err != nil {
		t.Errorf("can't find cephfs plugin by name")
	}

	cephfsSpec, err := plug.(*cephfsPlugin).ConstructVolumeSpec("cephfsVolume", "/cephfsVolume/")

	if cephfsSpec.Name() != "cephfsVolume" {
		t.Errorf("Get wrong cephfs spec name, got: %s", cephfsSpec.Name())
	}
}
