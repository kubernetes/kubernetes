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
	"crypto/sha256"
	"errors"
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1alpha1"
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
		return "", errors.New("missing spec")
	}

	if spec.PersistentVolume == nil {
		return "", errors.New("missing persistent volume")
	}

	//	namespace := spec.PersistentVolume.GetObjectMeta().GetNamespace()
	pvName := spec.PersistentVolume.GetName()
	attachID := fmt.Sprintf("pv-%s", hashAttachmentName(pvName, string(nodeName)))

	attachment := &storage.VolumeAttachment{
		ObjectMeta: meta.ObjectMeta{
			Name: attachID,
			//	Namespace: namespace, TODO should VolumeAttachment namespaced ?
		},
		Spec: storage.VolumeAttachmentSpec{
			NodeName: string(nodeName),
			Attacher: csiPluginName,
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
		},
	}
	attach, err := c.k8s.StorageV1alpha1().VolumeAttachments().Create(attachment)
	if err != nil {
		glog.Error(log("attacher.Attach failed: %v", err))
		return "", err
	}
	glog.V(4).Info(log("volume attachment sent: [%v]", attach.GetName()))

	return attach.GetName(), nil
}

func (c *csiAttacher) WaitForAttach(spec *volume.Spec, attachID string, pod *v1.Pod, timeout time.Duration) (string, error) {
	glog.V(4).Info(log("waiting for attachment update from CSI driver [attachment.ID=%v]", attachID))

	source, err := getCSISourceFromSpec(spec)
	if err != nil {
		glog.Error(log("attach.WaitForAttach failed to get volume source: %v", err))
		return "", err
	}

	ticker := time.NewTicker(c.waitSleepTime)
	defer ticker.Stop()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))
			attach, err := c.k8s.StorageV1alpha1().VolumeAttachments().Get(attachID, meta.GetOptions{})
			if err != nil {
				// log error, but continue to check again
				glog.Error(log("attacher.WaitForAttach failed (will continue to try): %v", err))
			}
			// attachment OK
			if attach.Status.Attached {
				return attachID, nil
			}
			// driver reports attach error
			attachErr := attach.Status.AttachError
			if attachErr != nil {
				glog.Error(log("attachment for %v failed: %v", source.VolumeHandle, attachErr.Message))
				return "", errors.New(attachErr.Message)
			}
		case <-timer.C:
			glog.Error(log("attacher.WaitForAttach timeout after %v [volume=%v; attachment.ID=%v]", timeout, source.VolumeHandle, attachID))
			return "", fmt.Errorf("attachment timeout for volume %v", source.VolumeHandle)
		}

	}
}

func (c *csiAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	return nil, errors.New("unimplemented")
}

func (c *csiAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	return "", errors.New("unimplemented")
}

func (c *csiAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	return errors.New("unimplemented")
}

var _ volume.Detacher = &csiAttacher{}

func (c *csiAttacher) Detach(deviceName string, nodeName types.NodeName) error {
	return errors.New("unimplemented")
}

func (c *csiAttacher) UnmountDevice(deviceMountPath string) error {
	return errors.New("unimplemented")
}

func hashAttachmentName(pvName, nodeName string) string {
	result := sha256.Sum256([]byte(fmt.Sprintf("%s%s", pvName, nodeName)))
	return fmt.Sprintf("%x", result)
}
