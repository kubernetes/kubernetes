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

package csi

import (
	"fmt"
	"os"
	"path"
	"testing"

	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// create a plugin mgr to load plugins and setup a fake client
func newTestPlugin(t *testing.T) (*csiPlugin, string) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHost(
		tmpDir,
		fakeClient,
		nil,
	)
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName(csiPluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", csiPluginName)
	}

	csiPlug, ok := plug.(*csiPlugin)
	if !ok {
		t.Fatalf("cannot assert plugin to be type csiPlugin")
	}

	return csiPlug, tmpDir
}

func makeTestPV(name string, sizeGig int, driverName, volID string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name:      name,
			Namespace: testns,
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(
					fmt.Sprintf("%dGi", sizeGig),
				),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       driverName,
					VolumeHandle: volID,
					ReadOnly:     false,
				},
			},
		},
	}
}

func TestPluginGetPluginName(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)
	if plug.GetPluginName() != "kubernetes.io/csi" {
		t.Errorf("unexpected plugin name %v", plug.GetPluginName())
	}
}

func TestPluginGetVolumeName(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)
	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	spec := volume.NewSpecFromPersistentVolume(pv, false)
	name, err := plug.GetVolumeName(spec)
	if err != nil {
		t.Fatalf("GetVolumeName failed: %v", err)
	}
	if name != testVol {
		t.Errorf("unexpected volume name %s", name)
	}
}

func TestPluginCanSupport(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	spec := volume.NewSpecFromPersistentVolume(pv, false)

	if !plug.CanSupport(spec) {
		t.Errorf("should support CSI spec")
	}
}

func TestPluginConstructVolumeSpec(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	dir := path.Join(tmpDir, fmt.Sprintf(
		"pods/%s/volumes/kubernetes.io~csi/%s/%s",
		testPodUID, testDriver, testVol,
	))

	// rebuild spec
	spec, err := plug.ConstructVolumeSpec(pv.Name, dir)
	if err != nil {
		t.Fatalf("ConstructVolumeSpec failed %v", err)
	}
	if spec.Name() != "test-pv" {
		t.Errorf("Unexpected spec name %s", spec.Name())
	}
	if spec.PersistentVolume.Spec.CSI.Driver != testDriver {
		t.Errorf("unexpected CSI.Driver %v", spec.PersistentVolume.Spec.CSI.Driver)
	}
}

func TestPluginNewMounter(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	mounter, err := plug.NewMounter(
		volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if mounter == nil {
		t.Fatal("failed to create CSI mounter")
	}
	csiMounter := mounter.(*csiMountMgr)

	// validate mounter fields
	if csiMounter.driverName != testDriver {
		t.Error("mounter driver name not set")
	}
	if csiMounter.volumeID != testVol {
		t.Error("mounter volume id not set")
	}
	if csiMounter.pod == nil {
		t.Error("mounter pod not set")
	}
	if csiMounter.podUID == types.UID("") {
		t.Error("mounter podUID mot set")
	}
}

func TestPluginNewUnmounter(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	unmounter, err := plug.NewUnmounter(pv.ObjectMeta.Name, testPodUID)
	csiUnmounter := unmounter.(*csiMountMgr)

	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if csiUnmounter == nil {
		t.Fatal("failed to create CSI mounter")
	}

	if csiUnmounter.podUID != testPodUID {
		t.Error("podUID not set")
	}
}

func TestPluginNewAttacher(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}

	csiAttacher := attacher.(*csiAttacher)
	if csiAttacher.plugin == nil {
		t.Error("plugin not set for attacher")
	}
	if csiAttacher.k8s == nil {
		t.Error("Kubernetes client not set for attacher")
	}
}
