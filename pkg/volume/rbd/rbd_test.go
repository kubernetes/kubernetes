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

package rbd

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/rbd" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

type fakeDiskManager struct {
	tmpDir string
}

func NewFakeDiskManager() *fakeDiskManager {
	return &fakeDiskManager{
		tmpDir: utiltesting.MkTmpdirOrDie("rbd_test"),
	}
}

func (fake *fakeDiskManager) Cleanup() {
	os.RemoveAll(fake.tmpDir)
}

func (fake *fakeDiskManager) MakeGlobalPDName(disk rbd) string {
	return fake.tmpDir
}
func (fake *fakeDiskManager) AttachDisk(b rbdMounter) error {
	globalPath := b.manager.MakeGlobalPDName(*b.rbd)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakeDiskManager) DetachDisk(c rbdUnmounter, mntPath string) error {
	globalPath := c.manager.MakeGlobalPDName(*c.rbd)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakeDiskManager) CreateImage(provisioner *rbdVolumeProvisioner) (r *v1.RBDVolumeSource, volumeSizeGB int, err error) {
	return nil, 0, fmt.Errorf("not implemented")
}

func (fake *fakeDiskManager) DeleteImage(deleter *rbdVolumeDeleter) error {
	return fmt.Errorf("not implemented")
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fdm := NewFakeDiskManager()
	defer fdm.Cleanup()
	exec := mount.NewFakeExec(nil)
	mounter, err := plug.(*rbdPlugin).newMounterInternal(spec, types.UID("poduid"), fdm, &mount.FakeMounter{}, exec, "secrets")
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Error("Got a nil Mounter")
	}

	path := mounter.GetPath()
	expectedPath := fmt.Sprintf("%s/pods/poduid/volumes/kubernetes.io~rbd/vol1", tmpDir)
	if path != expectedPath {
		t.Errorf("Unexpected path, expected %q, got: %q", expectedPath, path)
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

	unmounter, err := plug.(*rbdPlugin).newUnmounterInternal("vol1", types.UID("poduid"), fdm, &mount.FakeMounter{}, exec)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Error("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}

func TestPluginVolume(t *testing.T) {
	vol := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			RBD: &v1.RBDVolumeSource{
				CephMonitors: []string{"a", "b"},
				RBDImage:     "bar",
				FSType:       "ext4",
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol))
}
func TestPluginPersistentVolume(t *testing.T) {
	vol := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vol1",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				RBD: &v1.RBDVolumeSource{
					CephMonitors: []string{"a", "b"},
					RBDImage:     "bar",
					FSType:       "ext4",
				},
			},
		},
	}

	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
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
				RBD: &v1.RBDVolumeSource{
					CephMonitors: []string{"a", "b"},
					RBDImage:     "bar",
					FSType:       "ext4",
				},
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, client, nil))
	plug, _ := plugMgr.FindPluginByName(rbdPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}

func TestPersistAndLoadRBD(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testcases := []struct {
		rbdMounter               rbdMounter
		expectedJSONStr          string
		expectedLoadedRBDMounter rbdMounter
	}{
		{
			rbdMounter{},
			`{"Mon":null,"Id":"","Keyring":"","Secret":""}`,
			rbdMounter{},
		},
		{
			rbdMounter{
				rbd: &rbd{
					podUID:          "poduid",
					Pool:            "kube",
					Image:           "some-test-image",
					ReadOnly:        false,
					MetricsProvider: volume.NewMetricsStatFS("/tmp"),
				},
				Mon:     []string{"127.0.0.1"},
				Id:      "kube",
				Keyring: "",
				Secret:  "QVFEcTdKdFp4SmhtTFJBQUNwNDI3UnhGRzBvQ1Y0SUJwLy9pRUE9PQ==",
			},
			`
{
	"Pool": "kube",
	"Image": "some-test-image",
	"ReadOnly": false,
	"Mon": ["127.0.0.1"],
	"Id": "kube",
	"Keyring": "",
	"Secret": "QVFEcTdKdFp4SmhtTFJBQUNwNDI3UnhGRzBvQ1Y0SUJwLy9pRUE9PQ=="
}
			`,
			rbdMounter{
				rbd: &rbd{
					Pool:     "kube",
					Image:    "some-test-image",
					ReadOnly: false,
				},
				Mon:     []string{"127.0.0.1"},
				Id:      "kube",
				Keyring: "",
				Secret:  "QVFEcTdKdFp4SmhtTFJBQUNwNDI3UnhGRzBvQ1Y0SUJwLy9pRUE9PQ==",
			},
		},
	}

	util := &RBDUtil{}
	for _, c := range testcases {
		err = util.persistRBD(c.rbdMounter, tmpDir)
		if err != nil {
			t.Errorf("failed to persist rbd: %v, err: %v", c.rbdMounter, err)
		}
		jsonFile := filepath.Join(tmpDir, "rbd.json")
		jsonData, err := ioutil.ReadFile(jsonFile)
		if err != nil {
			t.Errorf("failed to read json file %s: %v", jsonFile, err)
		}
		if !assert.JSONEq(t, c.expectedJSONStr, string(jsonData)) {
			t.Errorf("json file does not match expected one: %s, should be %s", string(jsonData), c.expectedJSONStr)
		}
		tmpRBDMounter := rbdMounter{}
		err = util.loadRBD(&tmpRBDMounter, tmpDir)
		if err != nil {
			t.Errorf("faild to load rbd: %v", err)
		}
		if !reflect.DeepEqual(tmpRBDMounter, c.expectedLoadedRBDMounter) {
			t.Errorf("loaded rbd does not equal to expected one: %v, should be %v", tmpRBDMounter, c.rbdMounter)
		}
	}
}
