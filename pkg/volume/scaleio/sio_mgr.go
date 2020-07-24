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
	"strconv"

	"k8s.io/klog/v2"
	utilexec "k8s.io/utils/exec"

	siotypes "github.com/thecodeteam/goscaleio/types/v1"
)

type sioMgr struct {
	client     sioInterface
	configData map[string]string
	exec       utilexec.Interface
}

func newSioMgr(configs map[string]string, exec utilexec.Interface) (*sioMgr, error) {
	if configs == nil {
		return nil, errors.New("missing configuration data")
	}
	configs[confKey.protectionDomain] = defaultString(configs[confKey.protectionDomain], "default")
	configs[confKey.storagePool] = defaultString(configs[confKey.storagePool], "default")
	configs[confKey.sdcRootPath] = defaultString(configs[confKey.sdcRootPath], sdcRootPath)
	configs[confKey.storageMode] = defaultString(configs[confKey.storageMode], "ThinProvisioned")

	mgr := &sioMgr{configData: configs, exec: exec}
	return mgr, nil
}

// getClient safely returns an sioInterface
func (m *sioMgr) getClient() (sioInterface, error) {
	if m.client == nil {
		klog.V(4).Info(log("creating scaleio client"))
		configs := m.configData
		username := configs[confKey.username]
		password := configs[confKey.password]
		gateway := configs[confKey.gateway]
		b, err := strconv.ParseBool(configs[confKey.sslEnabled])
		if err != nil {
			klog.Error(log("failed to parse sslEnabled, must be either \"true\" or \"false\""))
			return nil, err
		}
		certsEnabled := b

		klog.V(4).Info(log("creating new client for gateway %s", gateway))
		client, err := newSioClient(gateway, username, password, certsEnabled, m.exec)
		if err != nil {
			klog.Error(log("failed to create scaleio client: %v", err))
			return nil, err
		}

		client.sysName = configs[confKey.system]
		client.pdName = configs[confKey.protectionDomain]
		client.spName = configs[confKey.storagePool]
		client.sdcPath = configs[confKey.sdcRootPath]
		client.provisionMode = configs[confKey.storageMode]
		client.sdcGUID = configs[confKey.sdcGUID]

		m.client = client

		klog.V(4).Info(log("client created successfully [gateway=%s]", gateway))
	}
	return m.client, nil
}

// CreateVolume creates a new ScaleIO volume
func (m *sioMgr) CreateVolume(volName string, sizeGB int64) (*siotypes.Volume, error) {
	client, err := m.getClient()
	if err != nil {
		return nil, err
	}

	klog.V(4).Infof("scaleio: creating volume %s", volName)
	vol, err := client.CreateVolume(volName, sizeGB)
	if err != nil {
		klog.V(4).Infof("scaleio: failed creating volume %s: %v", volName, err)
		return nil, err
	}
	klog.V(4).Infof("scaleio: created volume %s successfully", volName)
	return vol, nil
}

// AttachVolume maps a ScaleIO volume to the running node.  If flag multiMaps,
// ScaleIO will allow other SDC to map to volume.
func (m *sioMgr) AttachVolume(volName string, multipleMappings bool) (string, error) {
	client, err := m.getClient()
	if err != nil {
		klog.Error(log("attach volume failed: %v", err))
		return "", err
	}

	klog.V(4).Infoln(log("attaching volume %s", volName))
	iid, err := client.IID()
	if err != nil {
		klog.Error(log("failed to get instanceID"))
		return "", err
	}
	klog.V(4).Info(log("attaching volume %s to host instance %s", volName, iid))

	devs, err := client.Devs()
	if err != nil {
		return "", err
	}

	vol, err := client.FindVolume(volName)
	if err != nil {
		klog.Error(log("failed to find volume %s: %v", volName, err))
		return "", err
	}

	// handle vol if already attached
	if len(vol.MappedSdcInfo) > 0 {
		if m.isSdcMappedToVol(iid, vol) {
			klog.V(4).Info(log("skipping attachment, volume %s already attached to sdc %s", volName, iid))
			return devs[vol.ID], nil
		}
	}

	// attach volume, get deviceName
	if err := client.AttachVolume(sioVolumeID(vol.ID), multipleMappings); err != nil {
		klog.Error(log("attachment for volume %s failed :%v", volName, err))
		return "", err
	}
	device, err := client.WaitForAttachedDevice(vol.ID)
	if err != nil {
		klog.Error(log("failed while waiting for device to attach: %v", err))
		return "", err
	}
	klog.V(4).Info(log("volume %s attached successfully as %s to instance %s", volName, device, iid))
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
		klog.Error("scaleio: failed to get instanceID")
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
		klog.Error(log("failed to get instanceID: %v", err))
		return err
	}

	vol, err := client.FindVolume(volName)
	if err != nil {
		return err
	}
	if !m.isSdcMappedToVol(iid, vol) {
		klog.Warning(log(
			"skipping detached, vol %s not attached to instance %s",
			volName, iid,
		))
		return nil
	}

	if err := client.DetachVolume(sioVolumeID(vol.ID)); err != nil {
		klog.Error(log("failed to detach vol %s: %v", volName, err))
		return err
	}

	klog.V(4).Info(log("volume %s detached successfully", volName))

	return nil
}

// DeleteVolumes removes the ScaleIO volume
func (m *sioMgr) DeleteVolume(volName string) error {
	client, err := m.getClient()
	if err != nil {
		return err
	}

	vol, err := client.FindVolume(volName)
	if err != nil {
		return err
	}

	if err := client.DeleteVolume(sioVolumeID(vol.ID)); err != nil {
		klog.Error(log("failed to delete volume %s: %v", volName, err))
		return err
	}

	klog.V(4).Info(log("deleted volume %s successfully", volName))
	return nil

}

// isSdcMappedToVol returns true if the sdc is mapped to the volume
func (m *sioMgr) isSdcMappedToVol(sdcID string, vol *siotypes.Volume) bool {
	if len(vol.MappedSdcInfo) == 0 {
		klog.V(4).Info(log("no attachment found"))
		return false
	}

	for _, sdcInfo := range vol.MappedSdcInfo {
		if sdcInfo.SdcID == sdcID {
			return true
		}
	}
	return false
}
