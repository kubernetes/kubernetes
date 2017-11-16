/*
Copyright 2017 The Kubernetes Authors.

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

package scaleio

import (
	"fmt"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/golang/glog"

	api "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

var (
	testSioSystem  = "sio"
	testSioPD      = "default"
	testSioVol     = "vol-0001"
	testns         = "default"
	testSecret     = "sio-secret"
	testSioVolName = fmt.Sprintf("%s%s%s", testns, "-", testSioVol)
	podUID         = types.UID("sio-pod")
)

func newPluginMgr(t *testing.T, apiObject runtime.Object) (*volume.VolumePluginMgr, string) {
	tmpDir, err := utiltesting.MkTmpdir("scaleio-test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	fakeClient := fakeclient.NewSimpleClientset(apiObject)
	host := volumetest.NewFakeVolumeHostWithNodeLabels(
		tmpDir,
		fakeClient,
		nil,
		map[string]string{sdcGuidLabelName: "abc-123"},
	)
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	return plugMgr, tmpDir
}

func makeScaleIOSecret(name, namespace string) *api.Secret {
	return &api.Secret{
		ObjectMeta: meta.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			UID:       "1234567890",
		},
		Type: api.SecretType("kubernetes.io/scaleio"),
		Data: map[string][]byte{
			"username": []byte("username"),
			"password": []byte("password"),
		},
	}
}

func TestVolumeCanSupport(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t, makeScaleIOSecret(testSecret, testns))
	defer os.RemoveAll(tmpDir)
	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %s by name", sioPluginName)
	}
	if plug.GetPluginName() != "kubernetes.io/scaleio" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(
		&volume.Spec{
			Volume: &api.Volume{
				VolumeSource: api.VolumeSource{
					ScaleIO: &api.ScaleIOVolumeSource{},
				},
			},
		},
	) {
		t.Errorf("Expected true for CanSupport LibStorage VolumeSource")
	}
	if !plug.CanSupport(
		&volume.Spec{
			PersistentVolume: &api.PersistentVolume{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						ScaleIO: &api.ScaleIOPersistentVolumeSource{},
					},
				},
			},
		},
	) {
		t.Errorf("Expected true for CanSupport LibStorage PersistentVolumeSource")
	}
}

func TestVolumeGetAccessModes(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t, makeScaleIOSecret(testSecret, testns))
	defer os.RemoveAll(tmpDir)
	plug, err := plugMgr.FindPersistentPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", sioPluginName)
	}
	if !containsMode(plug.GetAccessModes(), api.ReadWriteOnce) {
		t.Errorf("Expected two AccessModeTypes:  %s or %s", api.ReadWriteOnce, api.ReadOnlyMany)
	}
}
func containsMode(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func TestVolumeMounterUnmounter(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t, makeScaleIOSecret(testSecret, testns))
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Errorf("Cannot assert plugin to be type sioPlugin")
	}

	vol := &api.Volume{
		Name: testSioVolName,
		VolumeSource: api.VolumeSource{
			ScaleIO: &api.ScaleIOVolumeSource{
				Gateway:          "http://test.scaleio:1111",
				System:           testSioSystem,
				ProtectionDomain: testSioPD,
				StoragePool:      "default",
				VolumeName:       testSioVol,
				FSType:           "ext4",
				SecretRef:        &api.LocalObjectReference{Name: testSecret},
				ReadOnly:         false,
			},
		},
	}

	sioMounter, err := sioPlug.NewMounter(
		volume.NewSpecFromVolume(vol),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if sioMounter == nil {
		t.Fatal("Got a nil Mounter")
	}

	sio := newFakeSio()
	sioVol := sioMounter.(*sioVolume)
	if err := sioVol.setSioMgr(); err != nil {
		t.Fatalf("failed to create sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	sioVol.sioMgr.CreateVolume(testSioVol, 8) //create vol ahead of time

	volPath := path.Join(tmpDir, fmt.Sprintf("pods/%s/volumes/kubernetes.io~scaleio/%s", podUID, testSioVolName))
	path := sioMounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := sioMounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	if sio.isMultiMap {
		t.Errorf("SetUp() - expecting multiple volume disabled by default")
	}

	// did we read sdcGuid label
	if _, ok := sioVol.sioMgr.configData[confKey.sdcGuid]; !ok {
		t.Errorf("Expected to find node label scaleio.sdcGuid, but did not find it")
	}

	// rebuild spec
	builtSpec, err := sioPlug.ConstructVolumeSpec(volume.NewSpecFromVolume(vol).Name(), path)
	if err != nil {
		t.Errorf("ConstructVolumeSpec failed %v", err)
	}
	if builtSpec.Name() != vol.Name {
		t.Errorf("Unexpected spec name %s", builtSpec.Name())
	}

	// unmount
	sioUnmounter, err := sioPlug.NewUnmounter(volume.NewSpecFromVolume(vol).Name(), podUID)
	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}
	if sioUnmounter == nil {
		t.Fatal("Got a nil Unmounter")
	}
	sioVol = sioUnmounter.(*sioVolume)
	if err := sioVol.resetSioMgr(); err != nil {
		t.Fatalf("failed to reset sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio

	if err := sioUnmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	// is mount point gone ?
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}
	// are we still mapped
	if sio.volume.MappedSdcInfo != nil {
		t.Errorf("expected SdcMappedInfo to be nil, volume may still be mapped")
	}
}

func TestVolumeProvisioner(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t, makeScaleIOSecret(testSecret, testns))
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Fatalf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Fatal("Cannot assert plugin to be type sioPlugin")
	}

	options := volume.VolumeOptions{
		ClusterName: "testcluster",
		PVC:         volumetest.CreateTestPVC("100Mi", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}
	options.PVC.Name = "testpvc"
	options.PVC.Namespace = testns

	options.PVC.Spec.AccessModes = []api.PersistentVolumeAccessMode{
		api.ReadOnlyMany,
	}

	options.Parameters = map[string]string{
		confKey.gateway:          "http://test.scaleio:11111",
		confKey.system:           "sio",
		confKey.protectionDomain: testSioPD,
		confKey.storagePool:      "default",
		confKey.secretName:       testSecret,
	}

	provisioner, err := sioPlug.NewProvisioner(options)
	if err != nil {
		t.Fatalf("failed to create new provisioner: %v", err)
	}
	if provisioner == nil {
		t.Fatal("got a nil provisioner")
	}
	sio := newFakeSio()
	sioVol := provisioner.(*sioVolume)
	if err := sioVol.setSioMgrFromConfig(); err != nil {
		t.Fatalf("failed to create scaleio mgr from config: %v", err)
	}
	sioVol.sioMgr.client = sio

	spec, err := provisioner.Provision()
	if err != nil {
		t.Fatalf("call to Provision() failed: %v", err)
	}

	if spec.Namespace != testns {
		t.Fatalf("unexpected namespace %v", spec.Namespace)
	}
	if spec.Spec.ScaleIO.SecretRef == nil {
		t.Fatalf("unexpected nil value for spec.SecretRef")
	}
	if spec.Spec.ScaleIO.SecretRef.Name != testSecret ||
		spec.Spec.ScaleIO.SecretRef.Namespace != testns {
		t.Fatalf("spec.SecretRef is not being set properly")
	}

	spec.Spec.ClaimRef = &api.ObjectReference{Namespace: testns}

	// validate provision
	actualSpecName := spec.Name
	actualVolName := spec.Spec.PersistentVolumeSource.ScaleIO.VolumeName
	if !strings.HasPrefix(actualSpecName, "k8svol-") {
		t.Errorf("expecting volume name to start with k8svol-, got %s", actualSpecName)
	}
	vol, err := sio.FindVolume(actualVolName)
	if err != nil {
		t.Fatalf("failed getting volume %v: %v", actualVolName, err)
	}
	if vol.Name != actualVolName {
		t.Errorf("expected volume name to be %s, got %s", actualVolName, vol.Name)
	}
	if vol.SizeInKb != 8*1024*1024 {
		glog.V(4).Info(log("unexpected volume size"))
	}

	// mount dynamic vol
	sioMounter, err := sioPlug.NewMounter(
		volume.NewSpecFromPersistentVolume(spec, false),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}
	sioVol = sioMounter.(*sioVolume)
	if err := sioVol.setSioMgr(); err != nil {
		t.Fatalf("failed to create sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	if err := sioMounter.SetUp(nil); err != nil {
		t.Fatalf("Expected success, got: %v", err)
	}

	// did we read sdcGuid label
	if _, ok := sioVol.sioMgr.configData[confKey.sdcGuid]; !ok {
		t.Errorf("Expected to find node label scaleio.sdcGuid, but did not find it")
	}

	// isMultiMap applied
	if !sio.isMultiMap {
		t.Errorf("SetUp()  expecting attached volume with multi-mapping")
	}

	// teardown dynamic vol
	sioUnmounter, err := sioPlug.NewUnmounter(spec.Name, podUID)
	if err != nil {
		t.Fatalf("Failed to make a new Unmounter: %v", err)
	}
	sioVol = sioUnmounter.(*sioVolume)
	if err := sioVol.resetSioMgr(); err != nil {
		t.Fatalf("failed to reset sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	if err := sioUnmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	// test deleter
	deleter, err := sioPlug.NewDeleter(volume.NewSpecFromPersistentVolume(spec, false))
	if err != nil {
		t.Fatalf("failed to create a deleter %v", err)
	}
	sioVol = deleter.(*sioVolume)
	if err := sioVol.setSioMgrFromSpec(); err != nil {
		t.Fatalf("failed to set sio mgr: %v", err)
	}
	sioVol.sioMgr.client = sio
	if err := deleter.Delete(); err != nil {
		t.Fatalf("failed while deleteing vol: %v", err)
	}
	path := deleter.GetPath()
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("Deleter did not delete path %v: %v", path, err)
	}
}

func TestVolumeProvisionerWithIncompleteConfig(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t, makeScaleIOSecret(testSecret, testns))
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Fatalf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Fatal("Cannot assert plugin to be type sioPlugin")
	}

	options := volume.VolumeOptions{
		ClusterName: "testcluster",
		PVName:      "pvc-sio-dynamic-vol",
		PVC:         volumetest.CreateTestPVC("100Mi", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}
	options.PVC.Namespace = testns

	options.PVC.Spec.AccessModes = []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}

	// incomplete options, test should fail
	_, err = sioPlug.NewProvisioner(options)
	if err == nil {
		t.Fatal("expected failure due to incomplete options")
	}
}

func TestVolumeProvisionerWithZeroCapacity(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t, makeScaleIOSecret(testSecret, testns))
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Fatalf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Fatal("Cannot assert plugin to be type sioPlugin")
	}

	options := volume.VolumeOptions{
		ClusterName: "testcluster",
		PVName:      "pvc-sio-dynamic-vol",
		PVC:         volumetest.CreateTestPVC("0Mi", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}
	options.PVC.Namespace = testns

	options.PVC.Spec.AccessModes = []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}

	options.Parameters = map[string]string{
		confKey.gateway:          "http://test.scaleio:11111",
		confKey.system:           "sio",
		confKey.protectionDomain: testSioPD,
		confKey.storagePool:      "default",
		confKey.secretName:       "sio-secret",
	}

	provisioner, _ := sioPlug.NewProvisioner(options)
	sio := newFakeSio()
	sioVol := provisioner.(*sioVolume)
	if err := sioVol.setSioMgrFromConfig(); err != nil {
		t.Fatalf("failed to create scaleio mgr from config: %v", err)
	}
	sioVol.sioMgr.client = sio

	_, err = provisioner.Provision()
	if err == nil {
		t.Fatalf("call to Provision() should fail with invalid capacity")
	}

}

func TestVolumeProvisionerWithSecretNamespace(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t, makeScaleIOSecret("sio-sec", "sio-ns"))
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Fatalf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Fatal("Cannot assert plugin to be type sioPlugin")
	}

	options := volume.VolumeOptions{
		ClusterName: "testcluster",
		PVName:      "pvc-sio-dynamic-vol",
		PVC:         volumetest.CreateTestPVC("100Mi", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}

	options.PVC.Spec.AccessModes = []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}

	options.PVC.Namespace = "pvc-ns"
	options.Parameters = map[string]string{
		confKey.gateway:          "http://test.scaleio:11111",
		confKey.system:           "sio",
		confKey.protectionDomain: testSioPD,
		confKey.storagePool:      "default",
		confKey.secretName:       "sio-sec",
		confKey.secretNamespace:  "sio-ns",
	}

	provisioner, _ := sioPlug.NewProvisioner(options)
	sio := newFakeSio()
	sioVol := provisioner.(*sioVolume)
	if err := sioVol.setSioMgrFromConfig(); err != nil {
		t.Fatalf("failed to create scaleio mgr from config: %v", err)
	}
	sioVol.sioMgr.client = sio

	spec, err := sioVol.Provision()
	if err != nil {
		t.Fatalf("call to Provision() failed: %v", err)
	}

	if spec.GetObjectMeta().GetNamespace() != "pvc-ns" {
		t.Fatalf("unexpected spec.namespace %s", spec.GetObjectMeta().GetNamespace())
	}

	if spec.Spec.ScaleIO.SecretRef.Name != "sio-sec" {
		t.Fatalf("unexpected spec.ScaleIOPersistentVolume.SecretRef.Name %v", spec.Spec.ScaleIO.SecretRef.Name)
	}

	if spec.Spec.ScaleIO.SecretRef.Namespace != "sio-ns" {
		t.Fatalf("unexpected spec.ScaleIOPersistentVolume.SecretRef.Namespace %v", spec.Spec.ScaleIO.SecretRef.Namespace)
	}
}
