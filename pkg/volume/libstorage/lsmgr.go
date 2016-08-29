/*
Copyright 2016 The Kubernetes Authors.

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

package libstorage

import (
	"fmt"

	"github.com/akutz/gofig"
	"github.com/golang/glog"

	lsctx "github.com/emccode/libstorage/api/context"
	lstypes "github.com/emccode/libstorage/api/types"
	lsutils "github.com/emccode/libstorage/api/utils"
	lsclient "github.com/emccode/libstorage/client"

	"k8s.io/kubernetes/pkg/volume"
)

type lsMgr interface {
	createVolume(*lsVolume) (*lstypes.Volume, error)
	attachVolume(*lsVolume) (string, error)
	detachVolume(*lsVolume) error
	deleteVolume(*lsVolume) error
	getService() string
	getHost() string
}

type libStorageMgr struct {
	instanceID string
	host       string
	service    string
	client     lstypes.Client
	ctx        lstypes.Context
	cfg        gofig.Config
}

func newLibStorageMgr(host, service string) *libStorageMgr {
	return &libStorageMgr{
		host:    host,
		service: service,
	}
}

func (m *libStorageMgr) getService() string {
	return m.service
}

func (m *libStorageMgr) getHost() string {
	return m.host
}

func (m *libStorageMgr) initMgr() {
	m.cfg = gofig.New()
	m.cfg.Set("libstorage.host", m.host)
	m.ctx = lsctx.Background()
	m.ctx = m.ctx.WithValue(lsctx.ServiceKey, m.service)
	glog.V(2).Infoln("LibStorage mgr init'd: host=%s, service=%", m.host, m.service)
}

// getClient safely returns a libstorage.Client
func (m *libStorageMgr) getClient() (lstypes.Client, error) {
	if m.client == nil {
		m.initMgr()
		client, err := lsclient.New(m.ctx, m.cfg)
		if err != nil {
			glog.Errorf("LibStorage: client init failed: %v", err)
			return nil, err
		}
		m.client = client
		return m.client, nil
	}
	return m.client, nil
}

// creates a new volume to represent spec
func (m *libStorageMgr) createVolume(lsVol *lsVolume) (*lstypes.Volume, error) {
	libClient, err := m.getClient()
	if err != nil {
		return nil, err
	}

	name := volume.GenerateVolumeName(lsVol.options.ClusterName, lsVol.options.PVName, 255)
	glog.V(4).Infof("libStorage: provisioning volume %v", name)

	volSizeBytes := lsVol.options.Capacity.Value()
	volSizeGB := int64(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))

	vol, err := libClient.Storage().VolumeCreate(
		m.ctx,
		name,
		&lstypes.VolumeCreateOpts{
			Size: &volSizeGB,
			Opts: lsutils.NewStore(),
		},
	)

	if err != nil {
		glog.Errorf("libStorage: failed to provision volume %s: %s", name, err)
		return nil, err
	}

	glog.V(4).Infof("libStorage: successfully provisioned volume %v", name)

	return vol, nil
}

// attach volume to host instance
func (m *libStorageMgr) attachVolume(lsVol *lsVolume) (string, error) {
	volName := lsVol.volName
	libClient, err := m.getClient()
	if err != nil {
		return "", err
	}

	instanceID, err := m.getInstanceID()
	if err != nil {
		return "", err
	}
	glog.V(4).Infof(
		"libStorage: attaching volume %v to host instace %s",
		volName, instanceID,
	)

	volByName, err := m.findVolumeByName(volName, false)
	if err != nil {
		glog.Errorf("libStorage: failed to find volume %v: %v",
			volName, err)
		return "", err
	}
	// retrieve vol and all its attachments
	vol, err := libClient.Storage().VolumeInspect(
		m.ctx,
		volByName.ID,
		&lstypes.VolumeInspectOpts{
			Attachments: true,
			Opts:        lsutils.NewStore(),
		},
	)
	// if already attached to instance host, done.
	if vol.Attachments != nil && len(vol.Attachments) > 0 {
		attach, err := m.findAttachmentByInstance(instanceID, vol.Attachments)
		if err != nil {
			return "", err
		}

		if attach != nil {
			glog.V(4).Infof(
				"libStorage: volume %v already attached as %v",
				vol.VolumeName(),
				attach.DeviceName,
			)
			return attach.DeviceName, nil
		}
	}

	// attach volume and get device path
	vol, attachToken, err := libClient.Storage().VolumeAttach(
		m.ctx,
		vol.ID,
		&lstypes.VolumeAttachOpts{
			Force: false,
			Opts:  lsutils.NewStore(),
		},
	)

	if err != nil {
		glog.Errorf(
			"libStorage: failed to attach volume %v: %v",
			lsVol.volName, err,
		)
		return "", err
	}

	// wait for device to show up in device list
	attached, result, err := libClient.Executor().WaitForDevice(
		m.ctx,
		&lstypes.WaitForDeviceOpts{
			Token:   attachToken,
			Timeout: attachTimeout,
		},
	)
	if err != nil {
		glog.Errorf(
			"libStorage: failed to wait for volume %s to attach",
			vol.Name)
		return "", err
	}
	if !attached {
		glog.V(4).Infof(
			"libStorage: volume %s not in attached device list",
			vol.Name)
		return "", fmt.Errorf("volume not in device list")
	}

	for token, device := range result.DeviceMap {
		if token == attachToken {
			glog.V(4).Infof(
				"libStorage: volume %v attached at device path %v",
				vol.Name, device,
			)
			return device, nil
		}
	}

	glog.Errorf("libStorage: failed to find attached device path")
	return "", fmt.Errorf("failed to get device path")
}

// detach volume from instance host
func (m *libStorageMgr) detachVolume(lsVol *lsVolume) error {
	volName := lsVol.volName
	libClient, err := m.getClient()
	if err != nil {
		return err
	}
	glog.V(4).Infof("libStorage: attempting to detach volume %v", volName)

	vol, err := m.findVolumeByName(volName, true)
	if err != nil {
		return err
	}
	_, err = libClient.Storage().VolumeDetach(
		m.ctx,
		vol.ID,
		&lstypes.VolumeDetachOpts{
			Force: false,
			Opts:  lsutils.NewStore(),
		},
	)

	if err != nil {
		glog.Error("libStorage: unable to detach volume ", volName)
		return err
	}

	glog.V(4).Info("libStorage: successfully detached volume %s", volName)

	return nil
}

func (m *libStorageMgr) deleteVolume(lsVol *lsVolume) error {
	volName := lsVol.volName
	libClient, err := m.getClient()
	if err != nil {
		return err
	}
	glog.V(4).Infof("libStorage: attempting to delete volume %v", volName)

	vol, err := m.findVolumeByName(volName, true)
	if err != nil {
		return err
	}
	err = libClient.Storage().VolumeRemove(
		m.ctx,
		vol.ID,
		lsutils.NewStore(),
	)

	if err != nil {
		glog.Errorf("libStorage: failed to delete volume %s: %v ",
			volName, err)
		return err
	}

	glog.V(4).Info("libStorage: deleted volume %s", volName)
	return nil
}

// returns volume by name
func (m *libStorageMgr) findVolumeByName(name string, attached bool) (*lstypes.Volume, error) {
	libClient, err := m.getClient()
	if err != nil || libClient == nil {
		return nil, err
	}

	vols, err := libClient.Storage().Volumes(
		m.ctx,
		&lstypes.VolumesOpts{Attachments: attached},
	)

	if err != nil {
		glog.Error("libStorage: storage.Volumes() failed: ", err)
		return nil, err
	}
	// get existing known vols
	if len(vols) < 1 {
		glog.V(4).Infof("libStorage: no volumes in storage system", name)
		return nil, fmt.Errorf("storage system has no volumes")
	}

	// search for vol with matching names
	for _, vol := range vols {
		if vol.Name == name {
			return vol, nil
		}
	}

	glog.Errorf("libStorage: volume %s not found", name)
	return nil, fmt.Errorf("volume not found")
}

func (m *libStorageMgr) findAttachmentByInstance(
	instanceID string,
	attachments []*lstypes.VolumeAttachment) (*lstypes.VolumeAttachment, error) {
	if attachments == nil || len(attachments) == 0 {
		return nil, fmt.Errorf("volume has no attachment")
	}

	for _, attach := range attachments {
		if instanceID == attach.InstanceID.ID {
			glog.V(4).Infof("libStorage: found attachment %s", attach.DeviceName)
			return attach, nil
		}
	}
	return nil, fmt.Errorf("attachment for instance not found")
}

func (m *libStorageMgr) getInstanceID() (string, error) {
	libClient, err := m.getClient()
	if err != nil {
		return "", err
	}

	if m.instanceID != "" {
		glog.V(4).Infof("libStorage: found instance id %s", m.instanceID)
		return m.instanceID, nil
	}

	id, err := libClient.Executor().InstanceID(m.ctx, lsutils.NewStore())
	if err != nil {
		glog.Errorf("libStorage: failed to get InstanceID: %v", err)
		return "", err
	}
	m.instanceID = id.ID
	return m.instanceID, nil
}
