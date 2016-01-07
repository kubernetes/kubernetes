/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package cinder

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "cinderTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/cinder" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{Cinder: &api.CinderVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}

	if !plug.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{Cinder: &api.CinderVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

type fakePDManager struct{}

func (fake *fakePDManager) AttachDisk(b *cinderVolumeBuilder, globalPDPath string) error {
	globalPath := makeGlobalPDName(b.plugin.host, b.pdName)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakePDManager) DetachDisk(c *cinderVolumeCleaner) error {
	globalPath := makeGlobalPDName(c.plugin.host, c.pdName)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakePDManager) CreateVolume(c *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, err error) {
	return "test-volume-name", 1, nil
}

func (fake *fakePDManager) DeleteVolume(cd *cinderVolumeDeleter) error {
	if cd.pdName != "test-volume-name" {
		return fmt.Errorf("Deleter got unexpected volume name: %s", cd.pdName)
	}
	return nil
}

func TestPlugin(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "cinderTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			Cinder: &api.CinderVolumeSource{
				VolumeID: "pd",
				FSType:   "ext4",
			},
		},
	}
	builder, err := plug.(*cinderPlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), &fakePDManager{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
	}
	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~cinder/vol1")
	path := builder.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := builder.SetUp(); err != nil {
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

	cleaner, err := plug.(*cinderPlugin).newCleanerInternal("vol1", types.UID("poduid"), &fakePDManager{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner: %v")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
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
	provisioner, err := plug.(*cinderPlugin).newProvisionerInternal(options, &fakePDManager{})
	persistentSpec, err := provisioner.NewPersistentVolumeTemplate()
	if err != nil {
		t.Errorf("NewPersistentVolumeTemplate() failed: %v", err)
	}

	// get 2nd Provisioner - persistent volume controller will do the same
	provisioner, err = plug.(*cinderPlugin).newProvisionerInternal(options, &fakePDManager{})
	err = provisioner.Provision(persistentSpec)
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.Cinder.VolumeID != "test-volume-name" {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.Cinder.VolumeID)
	}
	cap = persistentSpec.Spec.Capacity[api.ResourceStorage]
	size := cap.Value()
	if size != 1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*cinderPlugin).newDeleterInternal(volSpec, &fakePDManager{})
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}
