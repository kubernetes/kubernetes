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
	"errors"
	"testing"
	"time"

	siotypes "github.com/thecodeteam/goscaleio/types/v1"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/exec/testing"
)

var (
	fakeSdcID    = "test-sdc-123456789"
	fakeVolumeID = "1234567890"
	fakeDev      = "/dev/testABC"

	fakeConfig = map[string]string{
		confKey.gateway:    "http://sio.gateway:1234",
		confKey.sslEnabled: "false",
		confKey.system:     "scaleio",
		confKey.volumeName: "sio-0001",
		confKey.secretName: "sio-secret",
		confKey.username:   "c2lvdXNlcgo=",     // siouser
		confKey.password:   "c2lvcGFzc3dvcmQK", // siopassword
	}
)

func newTestMgr(t *testing.T) *sioMgr {
	host := volumetesting.NewFakeVolumeHost(t, "/tmp/fake", nil, nil)
	mgr, err := newSioMgr(fakeConfig, host, &testingexec.FakeExec{})
	if err != nil {
		t.Error(err)
	}
	mgr.client = newFakeSio()
	return mgr
}

func TestMgrNew(t *testing.T) {
	host := volumetesting.NewFakeVolumeHost(t, "/tmp/fake", nil, nil)
	mgr, err := newSioMgr(fakeConfig, host, &testingexec.FakeExec{})
	if err != nil {
		t.Fatal(err)
	}
	if mgr.configData == nil {
		t.Fatal("configuration data not set")
	}
	if mgr.configData[confKey.volumeName] != "sio-0001" {
		t.Errorf("expecting %s, got %s", "sio-0001", mgr.configData[confKey.volumeName])
	}

	// check defaults
	if mgr.configData[confKey.protectionDomain] != "default" {
		t.Errorf("unexpected value for confData[protectionDomain] %s", mgr.configData[confKey.protectionDomain])
	}
	if mgr.configData[confKey.storagePool] != "default" {
		t.Errorf("unexpected value for confData[storagePool] %s", mgr.configData[confKey.storagePool])
	}
	if mgr.configData[confKey.storageMode] != "ThinProvisioned" {
		t.Errorf("unexpected value for confData[storageMode] %s", mgr.configData[confKey.storageMode])
	}
}

func TestMgrGetClient(t *testing.T) {
	mgr := newTestMgr(t)
	_, err := mgr.getClient()
	if err != nil {
		t.Fatal(err)
	}
	if mgr.client == nil {
		t.Fatal("mgr.client not set")
	}
}

func TestMgrCreateVolume(t *testing.T) {
	mgr := newTestMgr(t)
	vol, err := mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	if err != nil {
		t.Fatal(err)
	}
	if vol.Name != "test-vol-0001" {
		t.Errorf("unexpected vol.Name %s", vol.Name)
	}
}

func TestMgrAttachVolume(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	device, err := mgr.AttachVolume("test-vol-0001", false)
	if err != nil {
		t.Fatal(err)
	}
	if device != "/dev/testABC" {
		t.Errorf("unexpected value for mapped device %s", device)
	}
}

func TestMgrAttachVolume_AlreadyAttached(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	mgr.AttachVolume("test-vol-0001", false)
	dev, err := mgr.AttachVolume("test-vol-0001", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dev != "/dev/testABC" {
		t.Errorf("unexpected value for mapped device %s", dev)
	}
}

func TestMgrAttachVolume_VolumeNotFoundError(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	_, err := mgr.AttachVolume("test-vol-0002", false)

	if err == nil {
		t.Error("attachVolume should fail with volume not found error")
	}
}

func TestMgrAttachVolume_WaitForAttachError(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	go func() {
		c := mgr.client.(*fakeSio)
		close(c.waitAttachCtrl)
	}()
	_, err := mgr.AttachVolume("test-vol-0001", false)
	if err == nil {
		t.Error("attachVolume should fail with attach timeout error")
	}
}

func TestMgrDetachVolume(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	mgr.AttachVolume("test-vol-0001", false)
	if err := mgr.DetachVolume("test-vol-0001"); err != nil {
		t.Fatal(err)
	}
	fakeSio := mgr.client.(*fakeSio)
	if len(fakeSio.volume.MappedSdcInfo) != 0 {
		t.Errorf("expecting attached sdc to 0, got %d", len(fakeSio.volume.MappedSdcInfo))
	}
	if len(fakeSio.devs) != 0 {
		t.Errorf("expecting local devs to be 0, got %d", len(fakeSio.devs))
	}

}
func TestMgrDetachVolume_VolumeNotFound(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	mgr.AttachVolume("test-vol-0001", false)
	err := mgr.DetachVolume("test-vol-0002")
	if err == nil {
		t.Fatal("expected a volume not found failure")
	}
}

func TestMgrDetachVolume_VolumeNotAttached(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	err := mgr.DetachVolume("test-vol-0001")
	if err != nil {
		t.Fatal(err)
	}
}

func TestMgrDetachVolume_VolumeAlreadyDetached(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	mgr.AttachVolume("test-vol-0001", false)
	mgr.DetachVolume("test-vol-0001")
	err := mgr.DetachVolume("test-vol-0001")
	if err != nil {
		t.Fatal("failed detaching a volume already detached")
	}
}

func TestMgrDetachVolume_WaitForDetachError(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	mgr.AttachVolume("test-vol-0001", false)
	err := mgr.DetachVolume("test-vol-0001")
	if err != nil {
		t.Error("detachVolume failed")
	}
}
func TestMgrDeleteVolume(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	err := mgr.DeleteVolume("test-vol-0001")
	if err != nil {
		t.Fatal(err)
	}
	sio := mgr.client.(*fakeSio)
	if sio.volume != nil {
		t.Errorf("volume not nil after delete operation")
	}
}
func TestMgrDeleteVolume_VolumeNotFound(t *testing.T) {
	mgr := newTestMgr(t)
	mgr.CreateVolume("test-vol-0001", 8*1024*1024)
	err := mgr.DeleteVolume("test-vol-0002")
	if err == nil {
		t.Fatal("expected volume not found error")
	}
}

// ************************************************************
// Helper Test Types
// ************************************************************
type fakeSio struct {
	volume         *siotypes.Volume
	waitAttachCtrl chan struct{}
	waitDetachCtrl chan struct{}
	devs           map[string]string
	isMultiMap     bool
}

func newFakeSio() *fakeSio {
	return &fakeSio{
		waitAttachCtrl: make(chan struct{}),
		waitDetachCtrl: make(chan struct{}),
	}
}

func (f *fakeSio) FindVolume(volumeName string) (*siotypes.Volume, error) {
	if f.volume == nil || f.volume.Name != volumeName {
		return nil, errors.New("volume not found")
	}
	return f.volume, nil
}

func (f *fakeSio) Volume(id sioVolumeID) (*siotypes.Volume, error) {
	if f.volume == nil || f.volume.ID != string(id) {
		return nil, errors.New("volume not found")
	}
	return f.volume, nil
}

func (f *fakeSio) CreateVolume(volName string, sizeGB int64) (*siotypes.Volume, error) {
	f.volume = &siotypes.Volume{
		ID:         fakeVolumeID,
		Name:       volName,
		SizeInKb:   int(sizeGB),
		VolumeType: "test",
	}

	return f.volume, nil
}

func (f *fakeSio) AttachVolume(id sioVolumeID, multiMaps bool) error {
	f.isMultiMap = multiMaps
	_, err := f.Volume(id)
	if err != nil {
		return err
	}
	f.volume.MappedSdcInfo = []*siotypes.MappedSdcInfo{
		{SdcID: fakeSdcID},
	}

	return nil
}

func (f *fakeSio) DetachVolume(id sioVolumeID) error {
	if _, err := f.Volume(id); err != nil {
		return err
	}
	f.volume.MappedSdcInfo = nil
	delete(f.devs, f.volume.ID)

	return nil
}

func (f *fakeSio) DeleteVolume(id sioVolumeID) error {
	if _, err := f.Volume(id); err != nil {
		return err
	}
	f.volume = nil
	return nil
}

func (f *fakeSio) IID() (string, error) {
	return fakeSdcID, nil
}

func (f *fakeSio) Devs() (map[string]string, error) {
	if f.volume == nil {
		return nil, errors.New("volume not found")
	}
	f.devs = map[string]string{
		f.volume.ID: fakeDev,
	}

	return f.devs, nil
}

func (f *fakeSio) GetVolumeRefs(volID sioVolumeID) (int, error) {
	if f.volume == nil {
		return 0, nil
	}
	return 1, nil
}

func (f *fakeSio) WaitForAttachedDevice(token string) (string, error) {
	select {
	case <-time.After(500 * time.Millisecond):
		return fakeDev, nil
	case <-f.waitAttachCtrl:
		return "", errors.New("attached device timeout")
	}
}

func (f *fakeSio) WaitForDetachedDevice(token string) error {
	select {
	case <-time.After(500 * time.Millisecond):
		delete(f.devs, f.volume.ID)
		return nil
	case <-f.waitDetachCtrl:
		return errors.New("detach device timeout")
	}
}
