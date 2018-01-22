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

package csi

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/volume"
)

type csiAttacher struct {
	plugin        *csiPlugin
	k8s           kubernetes.Interface
	waitSleepTime time.Duration
}

// volume.Attacher methods
var _ volume.Attacher = &csiAttacher{}

func (c *csiAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	if spec == nil {
		glog.Error(log("attacher.Attach missing volume.Spec"))
		return "", errors.New("missing spec")
	}

	csiSource, err := getCSISourceFromSpec(spec)
	if err != nil {
		glog.Error(log("attacher.Attach failed to get CSI persistent source: %v", err))
		return "", err
	}

	node := string(nodeName)
	pvName := spec.PersistentVolume.GetName()
	attachID := getAttachmentName(csiSource.VolumeHandle, csiSource.Driver, node)

	attachment := &storage.VolumeAttachment{
		ObjectMeta: meta.ObjectMeta{
			Name: attachID,
		},
		Spec: storage.VolumeAttachmentSpec{
			NodeName: node,
			Attacher: csiSource.Driver,
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
		},
		Status: storage.VolumeAttachmentStatus{Attached: false},
	}

	_, err = c.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
	alreadyExist := false
	if err != nil {
		if !apierrs.IsAlreadyExists(err) {
			glog.Error(log("attacher.Attach failed: %v", err))
			return "", err
		}
		alreadyExist = true
	}

	if alreadyExist {
		glog.V(4).Info(log("attachment [%v] for volume [%v] already exists (will not be recreated)", attachID, csiSource.VolumeHandle))
	} else {
		glog.V(4).Info(log("attachment [%v] for volume [%v] created successfully", attachID, csiSource.VolumeHandle))
	}

	// probe for attachment update here
	// NOTE: any error from waiting for attachment is logged only.  This is because
	// the primariy intent of the enclosing method is to create VolumeAttachment.
	// DONOT return that error here as it is mitigated in attacher.WaitForAttach.
	volAttachmentOK := true
	if _, err := c.waitForVolumeAttachment(csiSource.VolumeHandle, attachID, csiTimeout); err != nil {
		volAttachmentOK = false
		glog.Error(log("attacher.Attach attempted to wait for attachment to be ready, but failed with: %v", err))
	}

	glog.V(4).Info(log("attacher.Attach finished OK with VolumeAttachment verified=%t: attachment object [%s]", volAttachmentOK, attachID))

	return attachID, nil
}

func (c *csiAttacher) WaitForAttach(spec *volume.Spec, attachID string, pod *v1.Pod, timeout time.Duration) (string, error) {
	source, err := getCSISourceFromSpec(spec)
	if err != nil {
		glog.Error(log("attacher.WaitForAttach failed to extract CSI volume source: %v", err))
		return "", err
	}

	return c.waitForVolumeAttachment(source.VolumeHandle, attachID, timeout)
}

func (c *csiAttacher) waitForVolumeAttachment(volumeHandle, attachID string, timeout time.Duration) (string, error) {
	glog.V(4).Info(log("probing for updates from CSI driver for [attachment.ID=%v]", attachID))

	ticker := time.NewTicker(c.waitSleepTime)
	defer ticker.Stop()

	timer := time.NewTimer(timeout) // TODO (vladimirvivien) investigate making this configurable
	defer timer.Stop()

	//TODO (vladimirvivien) instead of polling api-server, change to a api-server watch
	for {
		select {
		case <-ticker.C:
			glog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))
			attach, err := c.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
			if err != nil {
				glog.Error(log("attacher.WaitForAttach failed (will continue to try): %v", err))
				continue
			}
			// if being deleted, fail fast
			if attach.GetDeletionTimestamp() != nil {
				glog.Error(log("VolumeAttachment [%s] has deletion timestamp, will not continue to wait for attachment", attachID))
				return "", errors.New("volume attachment is being deleted")
			}
			// attachment OK
			if attach.Status.Attached {
				return attachID, nil
			}
			// driver reports attach error
			attachErr := attach.Status.AttachError
			if attachErr != nil {
				glog.Error(log("attachment for %v failed: %v", volumeHandle, attachErr.Message))
				return "", errors.New(attachErr.Message)
			}
		case <-timer.C:
			glog.Error(log("attacher.WaitForAttach timeout after %v [volume=%v; attachment.ID=%v]", timeout, volumeHandle, attachID))
			return "", fmt.Errorf("attachment timeout for volume %v", volumeHandle)
		}
	}
}

func (c *csiAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	glog.V(4).Info(log("probing attachment status for %d volume(s) ", len(specs)))

	attached := make(map[*volume.Spec]bool)

	for _, spec := range specs {
		if spec == nil {
			glog.Error(log("attacher.VolumesAreAttached missing volume.Spec"))
			return nil, errors.New("missing spec")
		}
		source, err := getCSISourceFromSpec(spec)
		if err != nil {
			glog.Error(log("attacher.VolumesAreAttached failed: %v", err))
			continue
		}

		attachID := getAttachmentName(source.VolumeHandle, source.Driver, string(nodeName))
		glog.V(4).Info(log("probing attachment status for VolumeAttachment %v", attachID))
		attach, err := c.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
		if err != nil {
			glog.Error(log("attacher.VolumesAreAttached failed for attach.ID=%v: %v", attachID, err))
			continue
		}
		glog.V(4).Info(log("attacher.VolumesAreAttached attachment [%v] has status.attached=%t", attachID, attach.Status.Attached))
		attached[spec] = attach.Status.Attached
	}

	return attached, nil
}

func (c *csiAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	glog.V(4).Info(log("attacher.GetDeviceMountPath is not implemented"))
	return "", nil
}

func (c *csiAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	glog.V(4).Info(log("attacher.MountDevice is not implemented"))
	return nil
}

var _ volume.Detacher = &csiAttacher{}

func (c *csiAttacher) Detach(volumeName string, nodeName types.NodeName) error {
	// volumeName in format driverName<SEP>volumeHandle generated by plugin.GetVolumeName()
	if volumeName == "" {
		glog.Error(log("detacher.Detach missing value for parameter volumeName"))
		return errors.New("missing exepected parameter volumeName")
	}
	parts := strings.Split(volumeName, volNameSep)
	if len(parts) != 2 {
		glog.Error(log("detacher.Detach insufficient info encoded in volumeName"))
		return errors.New("volumeName missing expected data")
	}

	driverName := parts[0]
	volID := parts[1]
	attachID := getAttachmentName(volID, driverName, string(nodeName))
	if err := c.k8s.StorageV1beta1().VolumeAttachments().Delete(attachID, nil); err != nil {
		glog.Error(log("detacher.Detach failed to delete VolumeAttachment [%s]: %v", attachID, err))
		return err
	}

	glog.V(4).Info(log("detacher deleted ok VolumeAttachment.ID=%s", attachID))
	return c.waitForVolumeDetachment(volID, attachID)
}

func (c *csiAttacher) waitForVolumeDetachment(volumeHandle, attachID string) error {
	glog.V(4).Info(log("probing for updates from CSI driver for [attachment.ID=%v]", attachID))

	ticker := time.NewTicker(c.waitSleepTime)
	defer ticker.Stop()

	timeout := c.waitSleepTime * 10
	timer := time.NewTimer(timeout) // TODO (vladimirvivien) investigate making this configurable
	defer timer.Stop()

	//TODO (vladimirvivien) instead of polling api-server, change to a api-server watch
	for {
		select {
		case <-ticker.C:
			glog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))
			attach, err := c.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
			if err != nil {
				if apierrs.IsNotFound(err) {
					//object deleted or never existed, done
					glog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] not found, object deleted", attachID, volumeHandle))
					return nil
				}
				glog.Error(log("detacher.WaitForDetach failed for volume [%s] (will continue to try): %v", volumeHandle, err))
				continue
			}

			// driver reports attach error
			detachErr := attach.Status.DetachError
			if detachErr != nil {
				glog.Error(log("detachment for VolumeAttachment [%v] for volume [%s] failed: %v", attachID, volumeHandle, detachErr.Message))
				return errors.New(detachErr.Message)
			}
		case <-timer.C:
			glog.Error(log("detacher.WaitForDetach timeout after %v [volume=%v; attachment.ID=%v]", timeout, volumeHandle, attachID))
			return fmt.Errorf("detachment timed out for volume %v", volumeHandle)
		}
	}
}

func (c *csiAttacher) UnmountDevice(deviceMountPath string) error {
	glog.V(4).Info(log("detacher.UnmountDevice is not implemented"))
	return nil
}

// getAttachmentName returns csi-<sha252(volName,csiDriverName,NodeName>
func getAttachmentName(volName, csiDriverName, nodeName string) string {
	result := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", volName, csiDriverName, nodeName)))
	return fmt.Sprintf("csi-%x", result)
}
