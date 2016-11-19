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

package scaleio

import (
	"encoding/base64"
	"errors"
	"strconv"

	"github.com/golang/glog"

	siotypes "github.com/codedellemc/goscaleio/types/v1"
)

type storageInterface interface {
	CreateVolume(string, int64) (*siotypes.Volume, error)
	AttachVolume(string) (string, error)
	IsAttached(string) (bool, error)
	DetachVolume(string) error
	DeleteVolume(string) error
}

type sioMgr struct {
	client     sioInterface
	configData map[string]string
}

func newSioMgr(configs map[string]string) (*sioMgr, error) {
	if configs == nil {
		return nil, errors.New("missing configuration data")
	}
	configs[confKey.protectionDomain] = defaultString(configs[confKey.protectionDomain], "default")
	configs[confKey.storagePool] = defaultString(configs[confKey.storagePool], "default")
	configs[confKey.sdcRootPath] = defaultString(configs[confKey.sdcRootPath], "/opt/emc/scaleio/sdc/bin/")
	configs[confKey.storageMode] = defaultString(configs[confKey.storageMode], "ThinProvisioned")

	mgr := &sioMgr{configData: configs}
	return mgr, nil
}

// getClient safely returns an sioInterface
func (m *sioMgr) getClient() (sioInterface, error) {
	if m.client == nil {
		// decode username
		unEncoded := m.configData[confKey.username]
		unBin, err := base64.StdEncoding.DecodeString(unEncoded)
		if err != nil {
			return nil, err
		}
		username := string(unBin)

		// decode password
		pwdEncoded := m.configData[confKey.password]
		pwdBin, err := base64.StdEncoding.DecodeString(pwdEncoded)
		if err != nil {
			return nil, err
		}
		password := string(pwdBin)

		configs := m.configData
		gateway := configs[confKey.gateway]
		b, _ := strconv.ParseBool(configs[confKey.sslEnabled])
		certsEnabled := b

		client, err := newSioClient(gateway, username, password, certsEnabled)
		if err != nil {
			return nil, err
		}

		client.sysName = configs[confKey.system]
		client.pdName = configs[confKey.protectionDomain]
		client.spName = configs[confKey.storagePool]
		client.sdcPath = configs[confKey.sdcRootPath]
		client.provisionMode = configs[confKey.storageMode]

		m.client = client

		glog.V(4).Infof("sio: client created [gateway=%s]", gateway)
	}
	return m.client, nil
}

// CreateVolume creates a new ScaleIO volume
func (m *sioMgr) CreateVolume(volName string, sizeGB int64) (*siotypes.Volume, error) {
	client, err := m.getClient()
	if err != nil {
		return nil, err
	}

	glog.V(4).Infof("sio: creating volume %s", volName)
	vol, err := client.CreateVolume(volName, sizeGB)
	if err != nil {
		glog.V(4).Infof("sio: failed creating volume %s: %v", volName, err)
		return nil, err
	}
	glog.V(4).Infof("sio: created volume %s successfully", volName)
	return vol, nil
}

// AttachVolume maps a ScaleIO volume to the running node
func (m *sioMgr) AttachVolume(volName string) (string, error) {
	client, err := m.getClient()
	if err != nil {
		return "", err
	}

	iid, err := client.IID()
	if err != nil {
		glog.Errorf("sio: failed to get instanceID")
		return "", err
	}
	glog.V(4).Infof("sio: attaching volume %s to host instance %s", volName, iid)

	devs, err := client.Devs()
	if err != nil {
		return "", err
	}

	vol, err := client.FindVolume(volName)
	if err != nil {
		glog.Errorf("sio: failed to find volume %s: %v", volName, err)
		return "", err
	}

	if len(vol.MappedSdcInfo) > 0 {
		glog.V(4).Infof("sio: vol %s has attachments", volName)

		// if vol attached to current instance, done
		if m.isSdcMappedToVol(iid, vol) {
			glog.V(4).Infof("sio: skippping attachment, volume %s already attached to sdc %s", volName, iid)
			return devs[vol.ID], nil
		}

		// preempt attachment if vol attached to a different sdc
		glog.V(4).Infof("sio: preemting attachment for vol %s", volName)
		if err := client.DetachVolume(vol.ID); err != nil {
			glog.Errorf("sio: failed to preempt attachment for vol %s: %v", volName, err)
			return "", err
		}
	}

	// attach volume, get deviceName
	if err := client.AttachVolume(vol.ID); err != nil {
		glog.Errorf("sio: attachment for volume %s failed :%v", volName, err)
		return "", err
	}
	device, err := client.WaitForAttachedDevice(vol.ID)
	if err != nil {
		glog.Errorf("sio: failed while waiting for device to attach: %v", err)
		return "", err
	}
	glog.V(4).Infof("sio: volume %s attached succesfully as %s to instance %s", volName, device, iid)
	return device, nil
}

// IsAttached verifies that the named ScaleIO volume is still attached
func (m *sioMgr) IsAttached(volName string) (bool, error) {
	client, err := m.getClient()
	if err != nil {
		return false, err
	}
	iid, err := client.IID()
	if err != nil {
		glog.Errorf("sio: failed to get instanceID")
		return false, err
	}

	vol, err := client.FindVolume(volName)
	if err != nil {
		return false, err
	}
	return m.isSdcMappedToVol(iid, vol), nil
}

// DetachVolume detaches the name ScaleIO volume from an instance
func (m *sioMgr) DetachVolume(volName string) error {
	client, err := m.getClient()
	if err != nil {
		return err
	}
	iid, err := client.IID()
	if err != nil {
		glog.Errorf("sio: failed to get instanceID: %v", err)
		return err
	}

	vol, err := client.FindVolume(volName)
	if err != nil {
		return err
	}
	if !m.isSdcMappedToVol(iid, vol) {
		glog.Warningf(
			"sio: skipping detached, vol %s not attached to instance %s",
			volName, iid,
		)
		return nil
	}

	if err := client.DetachVolume(vol.ID); err != nil {
		glog.Errorf("sio: failed to detach vol %s: %v", volName, err)
		return err
	}
	if err := client.WaitForDetachedDevice(vol.ID); err != nil {
		glog.Errorf("sio: error waiting for vol %s to detach: %v", volName, err)
		return err
	}
	glog.V(4).Infof("sio: volume %s detached successfully", volName)
	return nil
}

// DeleteVolumes removes the ScaleIO volume
func (m *sioMgr) DeleteVolume(volName string) error {
	client, err := m.getClient()
	if err != nil {
		return err
	}
	iid, err := client.IID()
	if err != nil {
		glog.Errorf("sio: failed to get instanceID: %v", err)
		return err
	}

	vol, err := client.FindVolume(volName)
	if err != nil {
		return err
	}

	// if still attached, stop
	if m.isSdcMappedToVol(iid, vol) {
		glog.Errorf("sio: volume %s still attached,  unable to delete", volName)
		return errors.New("volume still attached")
	}

	if err := client.DeleteVolume(vol.ID); err != nil {
		glog.Errorf("sio: failed to delete volume %s: %v", volName, err)
		return err
	}

	glog.V(4).Infof("sio: deleted volume %s successfully", volName)
	return nil

}

//*****************************************************************
// Helpers
//*****************************************************************

// isSdcMappedToVol returns true if the sdc is mapped to the volume
func (m *sioMgr) isSdcMappedToVol(sdcID string, vol *siotypes.Volume) bool {
	if len(vol.MappedSdcInfo) == 0 {
		glog.V(4).Infof("sio: no attachment found")
		return false
	}

	for _, sdcInfo := range vol.MappedSdcInfo {
		if sdcInfo.SdcID == sdcID {
			return true
		}
	}
	return false
}
