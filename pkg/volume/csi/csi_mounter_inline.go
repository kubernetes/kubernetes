/*
Copyright 2018 The Kubernetes Authors.

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
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi/v0"
	"github.com/golang/glog"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	api "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

// setUpInline handles the provisioning and attachment of inline volumes embedded
// in a pod spec.  The method returns attachmentID, volumeHandle that was used
// during the inline setup.
//
// It follows these steps:
// 1. Check for volumeHandle
// - If not provided assume auto-provision, continue to provisioning #2
// - If provided, continue to attachment #3
// 2. Provisioning
// - Use volSpecName to initiate request to create vol with csi.CreateVolume()
// - return response.ID as volHandle
// 3. Attachment
// - Call csiPlugin.csiAttacher.Attach()
// - Use volHandle above to generate attachID
// - Wait for VolumeAttachment.ID from csiAttacher.Attach()
// - if attachment ok, return attachID.
func (c *csiMountMgr) setUpInline(csiSource *api.CSIVolumeSource) (string, string, error) {
	glog.V(4).Info(log("mounter.setupInline called for CSIVolumeSource"))

	if csiSource == nil {
		return "", "", errors.New("missing CSIVolumeSource")
	}

	var (
		driverName  = c.driverName
		volSpecName = c.spec.Name()
		namespace   = c.pod.Namespace
		volHandle   = csiSource.VolumeHandle
	)

	// missing volHandle means we should provision
	if volHandle == nil {
		glog.V(4).Info(log("mounter.setupInline No CSIVolumeSource.VolumeHandle provided, attempting to provision new volume"))
		vol, err := c.inlineProvision(volSpecName, namespace, c.spec.Volume)
		if err != nil {
			return "", "", err
		}
		handle := vol.Id
		volHandle = &handle
		c.volumeID = vol.Id
	}

	skip, err := c.plugin.skipAttach(driverName)
	if err != nil {
		glog.Error(log("mounter.setupInline failed to get attachability setting for driver: %v", err))
		return "", "", err
	}

	// trigger attachment and wait for attach.Name (if necessary)
	if skip {
		glog.V(4).Info(log("mounter.setupInline skipping volume attachment"))
		return "", *volHandle, nil
	}

	attachID, err := c.inlineAttach(*volHandle, csiSource, csiDefaultTimeout)
	if err != nil {
		return "", "", err
	}

	return attachID, *volHandle, nil
}

// inlineProvision will request the CSI driver to create a new volume.
// It will attempt to lookup referenced PVC, if one is provided, otherwise
// it will create a volume with default size of zero.
// Returns csi.Volume or error if failure.
func (c *csiMountMgr) inlineProvision(volSpecName, namespace string, volSource *api.Volume) (*csipb.Volume, error) {
	glog.V(4).Info(log("mounter.inlineProvision called for volume %s", volSpecName))

	if volSource == nil {
		return nil, errors.New("missing inline VolumeSource")
	}

	csiSource := volSource.CSI
	if csiSource == nil {
		return nil, errors.New("missing inline CSIVolumeSource")
	}

	pvcSource := volSource.PersistentVolumeClaim
	var volSizeBytes int64

	// if pvc volume source provided, retrieve associated capacity
	if pvcSource != nil {
		pvcName := pvcSource.ClaimName
		pvc, err := c.k8s.Core().PersistentVolumeClaims(namespace).Get(pvcName, meta.GetOptions{})

		if err != nil {
			if !apierrs.IsNotFound(err) {
				glog.Error(log("mounter.inlineProvision failed to get referenced PVC: %v", err))
				return nil, err
			}
			glog.V(4).Info(log("WARNING: mounter.inlineProvision PersistentVolumeClaim reference not found, setting capacity to zero."))
		} else {
			cap := pvc.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
			volSizeBytes = cap.Value()
			glog.V(4).Info(log("mounter.inlineProvision set capacity from referenced PVC [%d bytes]", volSizeBytes))
		}

	} else {
		glog.V(4).Info(log("WARNING: mounter.inlineProvision PersistentVolumeClaim reference not provided, capacity set to zero."))
	}

	// get controller pub secrets
	secrets := map[string]string{}
	if csiSource.ControllerPublishSecretRef != nil {
		name := csiSource.ControllerPublishSecretRef.Name
		sec, err := volutil.GetSecretForPod(c.pod, name, c.k8s)
		if err != nil {
			return nil, err
		}
		secrets = sec
	}

	// make controller pub request
	var volume *csipb.Volume
	err := wait.ExponentialBackoff(defaultBackoff(), func() (bool, error) {
		ctx, cancel := context.WithTimeout(context.Background(), csiDefaultTimeout)
		defer cancel()
		vol, createErr := c.csiClient.CreateVolume(ctx, volSpecName, volSizeBytes, secrets)
		if createErr == nil {
			volume = vol
			return true, nil
		}

		// can we recover
		if status, ok := status.FromError(createErr); ok {
			if status.Code() == codes.DeadlineExceeded {
				// CreateVolume timed out, give it another chance to complete
				glog.Warningf("Mounter.inlineProvision CreateVolume timed out, operation will be retried")
				return false, nil
			}
		}

		// CreateVolume failed , no reason to retry
		return false, createErr
	})

	if err != nil {
		glog.Errorf(log("mount.inlineProvision inline volume provision failed %s: %v", volSpecName, err))
		return nil, err
	}

	glog.V(4).Info(log("mounter.inlineProvision volume provisioned OK [Name: %s, ID:%s]", volSpecName, volume.Id))

	return volume, nil
}

// inlineAttach will create and post a VolumeAttachment API object
// to signal attachment to the external-attacher.  This subsequently causes
// the external-attacher to contact the CSI driver to create the attachment.
// Returns the ID for the VolumeAttachment API ovbject or an error if failure.
func (c *csiMountMgr) inlineAttach(volHandle string, csiSource *api.CSIVolumeSource, attachTimeout time.Duration) (string, error) {
	glog.V(4).Info(log("mounter.inlineAttach called for volumeHandle %s", volHandle))

	if csiSource == nil {
		return "", errors.New("missing inline CSIVolumeSource")
	}
	nodeName := string(c.plugin.host.GetNodeName())
	driverName := csiSource.Driver
	attachID := getAttachmentName(volHandle, driverName, nodeName)
	namespace := c.pod.Namespace
	volSource := c.spec.Volume

	attacher := &csiAttacher{
		k8s:           c.k8s,
		plugin:        c.plugin,
		waitSleepTime: 5 * time.Second,
	}

	attachment := &storage.VolumeAttachment{
		ObjectMeta: meta.ObjectMeta{
			Name: attachID,
		},
		Spec: storage.VolumeAttachmentSpec{
			NodeName: nodeName,
			Attacher: driverName,
			Source: storage.VolumeAttachmentSource{
				InlineVolumeSource: &storage.InlineVolumeSource{
					VolumeSource: volSource.VolumeSource,
					Namespace:    namespace,
				},
			},
		},
		Status: storage.VolumeAttachmentStatus{Attached: false},
	}

	// create and wait for attachment
	attachID, err := attacher.postVolumeAttachment(driverName, volHandle, attachment, attachTimeout)
	if err != nil {
		glog.Errorf(log("mount.inlineAttach failed to post attachment [attachID %s]: %v", attachID, err))
		return "", err
	}

	glog.V(4).Info(log("mounter.inlineAttach attached OK [driver:%s, volumeHandle: %s, attachID:%s]", driverName, volHandle, attachID))
	return attachID, nil
}

func generateVolHandle(prefix string, size int) string {
	return fmt.Sprintf("%s-%s", prefix, strings.Replace(string(uuid.NewUUID()), "-", "", -1)[0:size])
}
