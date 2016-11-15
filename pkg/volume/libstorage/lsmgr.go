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
	"errors"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/volume/libstorage/lstypes"
)

type lsMgr interface {
	createVolume(string, int64) (*lstypes.Volume, error)
	attachVolume(string) (string, error)
	isAttached(string) (bool, error)
	detachVolume(string) error
	deleteVolume(string) error
	getService() string
	getHost() string
}

type libStorageMgr struct {
	instanceID  string
	host        string
	service     string
	lsPluginDir string
	client      lstypes.Client
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

// getClient safely returns a libstorage.Client
func (m *libStorageMgr) getClient() (lstypes.Client, error) {
	if m.client == nil {
		glog.V(2).Infof("initializing libStorage client: host=%s, service=%s", m.host, m.service)
		client, err := newLsHttpClient(m.service, m.host)
		if err != nil {
			glog.Errorf("LibStorage: client init failed: %v", err)
			return nil, err
		}
		client.setExecDir(m.lsPluginDir)
		m.client = client
	}
	return m.client, nil
}

// creates a new volume to represent spec
func (m *libStorageMgr) createVolume(volName string, sizeGB int64) (*lstypes.Volume, error) {
	libClient, err := m.getClient()
	if err != nil {
		return nil, err
	}

	vol, err := libClient.CreateVolume(volName, sizeGB)
	if err != nil {
		glog.Errorf("libStorage: failed to provision volume %s: %s", volName, err)
		return nil, err
	}
	glog.V(4).Infof("libStorage: successfully provisioned volume %v", volName)
	return vol, nil
}

// attach volume to current pod  instance
func (m *libStorageMgr) attachVolume(volName string) (string, error) {
	libClient, err := m.getClient()
	if err != nil {
		return "", err
	}

	instanceID, err := libClient.IID()
	if err != nil {
		glog.Errorf("libStorage: failed to get InstanceID: %v", err)
		return "", err
	}

	glog.V(4).Infof(
		"libStorage: attaching volume %v to host with instanceID %v",
		volName, instanceID,
	)

	vol, err := libClient.FindVolume(volName)
	if err != nil {
		glog.Errorf("libStorage: failed to find volume %v: %v",
			volName, err)
		return "", err
	}

	// if already attached to instance host, done.
	if vol.Attachments != nil && len(vol.Attachments) > 0 {
		glog.V(4).Infof("libStorage: vol %s seems to be already attached", vol.Name)
		attach := m.findAttachmentByInstance(instanceID, vol.Attachments)
		if attach != nil {
			glog.Warningf(
				"libStorage: skipping attachment, volume %v already attached to instance as %v",
				vol.Name,
				attach.DeviceName,
			)
			return attach.DeviceName, nil
		}
		// preempt if attached device is attached elsewhere
		glog.V(4).Infof("libStorage: preempting attachment for vol %s", vol.Name)
		if err := m.detachVolume(volName); err != nil {
			glog.Errorf("libStorage: failed to preempt attachment: %v", err)
			return "", err
		}
	}

	// attach volume and get device path
	deviceToken, err := libClient.AttachVolume(vol.ID)
	if err != nil {
		glog.Errorf(
			"libStorage: attachment failed for volume %v: %v",
			volName, err,
		)
		return "", err
	}
	// wait for device to showup in instance device list
	device, err := libClient.WaitForAttachedDevice(deviceToken)
	if err != nil {
		glog.Errorf("libStorage: failed while waiting for device to attach: %v", err)
		return "", err
	}
	glog.V(4).Infof("libStorage: volume %s successfully attached as %s", volName, device)
	return device, nil
}

// isAttached returns true if the volume is attached
func (m *libStorageMgr) isAttached(volName string) (bool, error) {
	libClient, err := m.getClient()
	if err != nil {
		return false, err
	}
	vol, err := libClient.FindVolume(volName)
	if err != nil {
		return false, err
	}
	return (len(vol.Attachments) > 0), nil
}

// detach volume from instance host
func (m *libStorageMgr) detachVolume(volName string) error {
	libClient, err := m.getClient()
	if err != nil {
		return err
	}
	glog.V(4).Infof("libStorage: attempting to detach volume %v", volName)

	vol, err := libClient.FindVolume(volName)
	if err != nil {
		glog.Errorf("libStorage: failed to find volume %v: %v",
			volName, err)
		return err
	}
	if len(vol.Attachments) == 0 {
		glog.Warningf("libStorage: skipping detach, vol %s has no attachment", volName)
		return nil
	}

	// find attached device
	iid, err := libClient.IID()
	if err != nil {
		return err
	}
	var device string
	for _, attach := range vol.Attachments {
		if attach.InstanceID.ID == iid {
			device = attach.DeviceName
			break
		}
	}

	// detach operation
	_, err = libClient.DetachVolume(vol.ID)
	if err != nil {
		glog.Errorf("libStorage: detach operation for volume %s failed: %v", volName, err)
		return err
	}

	glog.V(4).Infof("ligStorage: waiting for detached device %s to be removed from instance", device)
	err = libClient.WaitForDetachedDevice(device)
	if err != nil {
		glog.Errorf("libStorage: timed out waiting for volume %v to detach", vol.Name)
		return err
	}

	glog.V(4).Info("libStorage: successfully detached volume ", volName)

	return nil
}

func (m *libStorageMgr) deleteVolume(volName string) error {
	libClient, err := m.getClient()
	if err != nil {
		return err
	}
	glog.V(4).Infof("libStorage: attempting to delete volume %v", volName)

	errVolNotFound := errors.New("volume not found")
	vol, err := libClient.FindVolume(volName)
	if err != nil {
		if err == errVolNotFound {
			glog.Warningf("libStorage: aborting delete, volume %s not found", volName)
			return nil
		} else {
			return err
		}
	}

	err = libClient.DeleteVolume(vol.ID)

	if err != nil {
		glog.Errorf("libStorage: failed to delete volume %s: %v ",
			volName, err)
		return err
	}

	glog.V(4).Info("libStorage: deleted volume %s successfully", volName)
	return nil
}

// findAttachmentByInstance returns a list of attachments for specified instance
func (m *libStorageMgr) findAttachmentByInstance(
	instanceID string,
	attachments []*lstypes.Attachment) *lstypes.Attachment {
	if attachments == nil || len(attachments) == 0 {
		glog.V(4).Infof("libStorage: volume has no attachments")
		return nil
	}

	for _, attach := range attachments {
		if instanceID == attach.InstanceID.ID {
			glog.V(4).Infof(
				"libStorage: found attachment %s for instance %s",
				attach.DeviceName, instanceID)
			return attach
		}
	}
	glog.V(4).Infof("libStorage: no attachment found for %v", instanceID)
	return nil
}
