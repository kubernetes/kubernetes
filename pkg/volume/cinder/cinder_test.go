// +build !providerless

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

package cinder

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/mount-utils"

	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/legacy-cloud-providers/openstack"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cinderTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/cinder" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{Cinder: &v1.CinderVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}

	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{Cinder: &v1.CinderPersistentVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

type fakePDManager struct {
	// How long should AttachDisk/DetachDisk take - we need slower AttachDisk in a test.
	attachDetachDuration time.Duration
}

func getFakeDeviceName(host volume.VolumeHost, pdName string) string {
	return filepath.Join(host.GetPluginDir(cinderVolumePluginName), "device", pdName)
}

// Real Cinder AttachDisk attaches a cinder volume. If it is not yet mounted,
// it mounts it to globalPDPath.
// We create a dummy directory (="device") and bind-mount it to globalPDPath
func (fake *fakePDManager) AttachDisk(b *cinderVolumeMounter, globalPDPath string) error {
	globalPath := makeGlobalPDName(b.plugin.host, b.pdName)
	fakeDeviceName := getFakeDeviceName(b.plugin.host, b.pdName)
	err := os.MkdirAll(fakeDeviceName, 0750)
	if err != nil {
		return err
	}
	// Attaching a Cinder volume can be slow...
	time.Sleep(fake.attachDetachDuration)

	// The volume is "attached", bind-mount it if it's not mounted yet.
	notmnt, err := b.mounter.IsLikelyNotMountPoint(globalPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(globalPath, 0750); err != nil {
				return err
			}
			notmnt = true
		} else {
			return err
		}
	}
	if notmnt {
		err = b.mounter.MountSensitiveWithoutSystemd(fakeDeviceName, globalPath, "", []string{"bind"}, nil)
		if err != nil {
			return err
		}
	}
	return nil
}

func (fake *fakePDManager) DetachDisk(c *cinderVolumeUnmounter) error {
	globalPath := makeGlobalPDName(c.plugin.host, c.pdName)
	fakeDeviceName := getFakeDeviceName(c.plugin.host, c.pdName)
	// unmount the bind-mount - should be fast
	err := c.mounter.Unmount(globalPath)
	if err != nil {
		return err
	}

	// "Detach" the fake "device"
	err = os.RemoveAll(fakeDeviceName)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakePDManager) CreateVolume(c *cinderVolumeProvisioner, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (volumeID string, volumeSizeGB int, labels map[string]string, fstype string, err error) {
	labels = make(map[string]string)
	labels[v1.LabelFailureDomainBetaZone] = "nova"
	return "test-volume-name", 1, labels, "", nil
}

func (fake *fakePDManager) DeleteVolume(cd *cinderVolumeDeleter) error {
	if cd.pdName != "test-volume-name" {
		return fmt.Errorf("Deleter got unexpected volume name: %s", cd.pdName)
	}
	return nil
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cinderTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			Cinder: &v1.CinderVolumeSource{
				VolumeID: "pd",
				FSType:   "ext4",
			},
		},
	}
	mounter, err := plug.(*cinderPlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), &fakePDManager{0}, mount.NewFakeMounter(nil))
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volPath := filepath.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~cinder/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	unmounter, err := plug.(*cinderPlugin).newUnmounterInternal("vol1", types.UID("poduid"), &fakePDManager{0}, mount.NewFakeMounter(nil))
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}

	// Test Provisioner
	options := volume.VolumeOptions{
		PVC:                           volumetest.CreateTestPVC("100Mi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}
	provisioner, err := plug.(*cinderPlugin).newProvisionerInternal(options, &fakePDManager{0})
	if err != nil {
		t.Errorf("ProvisionerInternal() failed: %v", err)
	}
	persistentSpec, err := provisioner.Provision(nil, nil)
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.Cinder.VolumeID != "test-volume-name" {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.Cinder.VolumeID)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// check nodeaffinity members
	if persistentSpec.Spec.NodeAffinity == nil {
		t.Errorf("Provision() returned unexpected nil NodeAffinity")
	}

	if persistentSpec.Spec.NodeAffinity.Required == nil {
		t.Errorf("Provision() returned unexpected nil NodeAffinity.Required")
	}

	n := len(persistentSpec.Spec.NodeAffinity.Required.NodeSelectorTerms)
	if n != 1 {
		t.Errorf("Provision() returned unexpected number of NodeSelectorTerms %d. Expected %d", n, 1)
	}

	n = len(persistentSpec.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions)
	if n != 1 {
		t.Errorf("Provision() returned unexpected number of MatchExpressions %d. Expected %d", n, 1)
	}

	req := persistentSpec.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions[0]

	if req.Key != v1.LabelFailureDomainBetaZone {
		t.Errorf("Provision() returned unexpected requirement key in NodeAffinity %v", req.Key)
	}

	if req.Operator != v1.NodeSelectorOpIn {
		t.Errorf("Provision() returned unexpected requirement operator in NodeAffinity %v", req.Operator)
	}

	if len(req.Values) != 1 || req.Values[0] != "nova" {
		t.Errorf("Provision() returned unexpected requirement value in NodeAffinity %v", req.Values)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*cinderPlugin).newDeleterInternal(volSpec, &fakePDManager{0})
	if err != nil {
		t.Errorf("DeleterInternal() failed: %v", err)
	}
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}

func TestGetVolumeLimit(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cinderTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	cloud, err := getOpenstackCloudProvider()
	if err != nil {
		t.Fatalf("can not instantiate openstack cloudprovider : %v", err)
	}

	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	volumeHost := volumetest.NewFakeVolumeHostWithCloudProvider(t, tmpDir, nil, nil, cloud)
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumeHost)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder")
	if err != nil {
		t.Fatalf("Can't find the plugin by name")
	}
	attachablePlugin, ok := plug.(volume.VolumePluginWithAttachLimits)
	if !ok {
		t.Fatalf("plugin %s is not of attachable type", plug.GetPluginName())
	}

	limits, err := attachablePlugin.GetVolumeLimits()
	if err != nil {
		t.Errorf("error fetching limits : %v", err)
	}
	if len(limits) == 0 {
		t.Fatalf("expecting limit from openstack got none")
	}
	limit, _ := limits[util.CinderVolumeLimitKey]
	if limit != 10 {
		t.Fatalf("expected volume limit to be 10 got %d", limit)
	}
}

func getOpenstackCloudProvider() (*openstack.OpenStack, error) {
	cfg := getOpenstackConfig()
	return openstack.NewFakeOpenStackCloud(cfg)
}

func getOpenstackConfig() openstack.Config {
	cfg := openstack.Config{
		Global: struct {
			AuthURL         string `gcfg:"auth-url"`
			Username        string
			UserID          string `gcfg:"user-id"`
			Password        string `datapolicy:"password"`
			TenantID        string `gcfg:"tenant-id"`
			TenantName      string `gcfg:"tenant-name"`
			TrustID         string `gcfg:"trust-id"`
			DomainID        string `gcfg:"domain-id"`
			DomainName      string `gcfg:"domain-name"`
			Region          string
			CAFile          string `gcfg:"ca-file"`
			SecretName      string `gcfg:"secret-name"`
			SecretNamespace string `gcfg:"secret-namespace"`
			KubeconfigPath  string `gcfg:"kubeconfig-path"`
		}{
			Username:   "user",
			Password:   "pass",
			TenantID:   "foobar",
			DomainID:   "2a73b8f597c04551a0fdc8e95544be8a",
			DomainName: "local",
			AuthURL:    "http://auth.url",
			UserID:     "user",
		},
		BlockStorage: openstack.BlockStorageOpts{
			NodeVolumeAttachLimit: 10,
		},
	}
	return cfg
}
