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
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	persistentVolumeInGlobalPath = "pv"
	globalMountInGlobalPath      = "globalmount"
)

type csiAttacher struct {
	plugin        *csiPlugin
	k8s           kubernetes.Interface
	waitSleepTime time.Duration

	csiClient csiClient
}

type verifyAttachDetachStatus func(attach *storage.VolumeAttachment, volumeHandle string) (bool, error)

// volume.Attacher methods
var _ volume.Attacher = &csiAttacher{}

var _ volume.Detacher = &csiAttacher{}

var _ volume.DeviceMounter = &csiAttacher{}

func (c *csiAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	if spec == nil {
		klog.Error(log("attacher.Attach missing volume.Spec"))
		return "", errors.New("missing spec")
	}

	pvSrc, err := getPVSourceFromSpec(spec)
	if err != nil {
		return "", errors.New(log("attacher.Attach failed to get CSIPersistentVolumeSource: %v", err))
	}

	node := string(nodeName)
	attachID := getAttachmentName(pvSrc.VolumeHandle, pvSrc.Driver, node)

	var vaSrc storage.VolumeAttachmentSource
	if spec.InlineVolumeSpecForCSIMigration {
		// inline PV scenario - use PV spec to populate VA source.
		// The volume spec will be populated by CSI translation API
		// for inline volumes. This allows fields required by the CSI
		// attacher such as AccessMode and MountOptions (in addition to
		// fields in the CSI persistent volume source) to be populated
		// as part of CSI translation for inline volumes.
		vaSrc = storage.VolumeAttachmentSource{
			InlineVolumeSpec: &spec.PersistentVolume.Spec,
		}
	} else {
		// regular PV scenario - use PV name to populate VA source
		pvName := spec.PersistentVolume.GetName()
		vaSrc = storage.VolumeAttachmentSource{
			PersistentVolumeName: &pvName,
		}
	}

	attachment := &storage.VolumeAttachment{
		ObjectMeta: meta.ObjectMeta{
			Name: attachID,
		},
		Spec: storage.VolumeAttachmentSpec{
			NodeName: node,
			Attacher: pvSrc.Driver,
			Source:   vaSrc,
		},
	}

	_, err = c.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
	alreadyExist := false
	if err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return "", errors.New(log("attacher.Attach failed: %v", err))
		}
		alreadyExist = true
	}

	if alreadyExist {
		klog.V(4).Info(log("attachment [%v] for volume [%v] already exists (will not be recreated)", attachID, pvSrc.VolumeHandle))
	} else {
		klog.V(4).Info(log("attachment [%v] for volume [%v] created successfully", attachID, pvSrc.VolumeHandle))
	}

	if _, err := c.waitForVolumeAttachment(pvSrc.VolumeHandle, attachID, csiTimeout); err != nil {
		return "", err
	}

	klog.V(4).Info(log("attacher.Attach finished OK with VolumeAttachment object [%s]", attachID))

	// Don't return attachID as a devicePath. We can reconstruct the attachID using getAttachmentName()
	return "", nil
}

func (c *csiAttacher) WaitForAttach(spec *volume.Spec, _ string, pod *v1.Pod, timeout time.Duration) (string, error) {
	source, err := getPVSourceFromSpec(spec)
	if err != nil {
		return "", errors.New(log("attacher.WaitForAttach failed to extract CSI volume source: %v", err))
	}

	attachID := getAttachmentName(source.VolumeHandle, source.Driver, string(c.plugin.host.GetNodeName()))

	return c.waitForVolumeAttachment(source.VolumeHandle, attachID, timeout)
}

func (c *csiAttacher) waitForVolumeAttachment(volumeHandle, attachID string, timeout time.Duration) (string, error) {
	klog.V(4).Info(log("probing for updates from CSI driver for [attachment.ID=%v]", attachID))

	timer := time.NewTimer(timeout) // TODO (vladimirvivien) investigate making this configurable
	defer timer.Stop()

	return c.waitForVolumeAttachmentInternal(volumeHandle, attachID, timer, timeout)
}

func (c *csiAttacher) waitForVolumeAttachmentInternal(volumeHandle, attachID string, timer *time.Timer, timeout time.Duration) (string, error) {

	klog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))
	attach, err := c.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), attachID, meta.GetOptions{})
	if err != nil {
		klog.Error(log("attacher.WaitForAttach failed for volume [%s] (will continue to try): %v", volumeHandle, err))
		return "", fmt.Errorf("volume %v has GET error for volume attachment %v: %v", volumeHandle, attachID, err)
	}
	err = c.waitForVolumeAttachDetachStatus(attach, volumeHandle, attachID, timer, timeout, verifyAttachmentStatus)
	if err != nil {
		return "", err
	}
	return attach.Name, nil
}

func (c *csiAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	klog.V(4).Info(log("probing attachment status for %d volume(s) ", len(specs)))

	attached := make(map[*volume.Spec]bool)

	for _, spec := range specs {
		if spec == nil {
			klog.Error(log("attacher.VolumesAreAttached missing volume.Spec"))
			return nil, errors.New("missing spec")
		}
		pvSrc, err := getPVSourceFromSpec(spec)
		if err != nil {
			attached[spec] = false
			klog.Error(log("attacher.VolumesAreAttached failed to get CSIPersistentVolumeSource: %v", err))
			continue
		}
		driverName := pvSrc.Driver
		volumeHandle := pvSrc.VolumeHandle

		skip, err := c.plugin.skipAttach(driverName)
		if err != nil {
			klog.Error(log("Failed to check CSIDriver for %s: %s", driverName, err))
		} else {
			if skip {
				// This volume is not attachable, pretend it's attached
				attached[spec] = true
				continue
			}
		}

		attachID := getAttachmentName(volumeHandle, driverName, string(nodeName))
		var attach *storage.VolumeAttachment
		if c.plugin.volumeAttachmentLister != nil {
			attach, err = c.plugin.volumeAttachmentLister.Get(attachID)
			if err == nil {
				attached[spec] = attach.Status.Attached
				continue
			}
			klog.V(4).Info(log("attacher.VolumesAreAttached failed in AttachmentLister for attach.ID=%v: %v. Probing the API server.", attachID, err))
		}
		// The cache lookup is not setup or the object is not found in the cache.
		// Get the object from the API server.
		klog.V(4).Info(log("probing attachment status for VolumeAttachment %v", attachID))
		attach, err = c.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), attachID, meta.GetOptions{})
		if err != nil {
			attached[spec] = false
			klog.Error(log("attacher.VolumesAreAttached failed for attach.ID=%v: %v", attachID, err))
			continue
		}
		klog.V(4).Info(log("attacher.VolumesAreAttached attachment [%v] has status.attached=%t", attachID, attach.Status.Attached))
		attached[spec] = attach.Status.Attached
	}

	return attached, nil
}

func (c *csiAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	klog.V(4).Info(log("attacher.GetDeviceMountPath(%v)", spec))
	deviceMountPath, err := makeDeviceMountPath(c.plugin, spec)
	if err != nil {
		return "", errors.New(log("attacher.GetDeviceMountPath failed to make device mount path: %v", err))
	}
	klog.V(4).Infof("attacher.GetDeviceMountPath succeeded, deviceMountPath: %s", deviceMountPath)
	return deviceMountPath, nil
}

func (c *csiAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	klog.V(4).Infof(log("attacher.MountDevice(%s, %s)", devicePath, deviceMountPath))

	if deviceMountPath == "" {
		return errors.New(log("attacher.MountDevice failed, deviceMountPath is empty"))
	}

	corruptedDir := false
	mounted, err := isDirMounted(c.plugin, deviceMountPath)
	if err != nil {
		klog.Error(log("attacher.MountDevice failed while checking mount status for dir [%s]", deviceMountPath))
		if isCorruptedDir(deviceMountPath) {
			corruptedDir = true // leave to CSI driver to handle corrupted mount
			klog.Warning(log("attacher.MountDevice detected corrupted mount for dir [%s]", deviceMountPath))
		} else {
			return err
		}
	}

	if mounted && !corruptedDir {
		klog.V(4).Info(log("attacher.MountDevice skipping mount, dir already mounted [%s]", deviceMountPath))
		return nil
	}

	// Setup
	if spec == nil {
		return errors.New(log("attacher.MountDevice failed, spec is nil"))
	}
	csiSource, err := getPVSourceFromSpec(spec)
	if err != nil {
		return errors.New(log("attacher.MountDevice failed to get CSIPersistentVolumeSource: %v", err))
	}

	// lets check if node/unstage is supported
	if c.csiClient == nil {
		c.csiClient, err = newCsiDriverClient(csiDriverName(csiSource.Driver))
		if err != nil {
			return errors.New(log("attacher.MountDevice failed to create newCsiDriverClient: %v", err))
		}
	}
	csi := c.csiClient

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()
	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := csi.NodeSupportsStageUnstage(ctx)
	if err != nil {
		return err
	}

	// Get secrets and publish context required for mountDevice
	nodeName := string(c.plugin.host.GetNodeName())
	publishContext, err := c.plugin.getPublishContext(c.k8s, csiSource.VolumeHandle, csiSource.Driver, nodeName)

	if err != nil {
		return volumetypes.NewTransientOperationFailure(err.Error())
	}

	nodeStageSecrets := map[string]string{}
	// we only require secrets if csiSource has them and volume has NodeStage capability
	if csiSource.NodeStageSecretRef != nil && stageUnstageSet {
		nodeStageSecrets, err = getCredentialsFromSecret(c.k8s, csiSource.NodeStageSecretRef)
		if err != nil {
			err = fmt.Errorf("fetching NodeStageSecretRef %s/%s failed: %v",
				csiSource.NodeStageSecretRef.Namespace, csiSource.NodeStageSecretRef.Name, err)
			// if we failed to fetch secret then that could be a transient error
			return volumetypes.NewTransientOperationFailure(err.Error())
		}
	}

	// Store volume metadata for UnmountDevice. Keep it around even if the
	// driver does not support NodeStage, UnmountDevice still needs it.
	if err = os.MkdirAll(deviceMountPath, 0750); err != nil && !corruptedDir {
		return errors.New(log("attacher.MountDevice failed to create dir %#v:  %v", deviceMountPath, err))
	}
	klog.V(4).Info(log("created target path successfully [%s]", deviceMountPath))
	dataDir := filepath.Dir(deviceMountPath)
	data := map[string]string{
		volDataKey.volHandle:  csiSource.VolumeHandle,
		volDataKey.driverName: csiSource.Driver,
	}
	if err = saveVolumeData(dataDir, volDataFileName, data); err != nil {
		klog.Error(log("failed to save volume info data: %v", err))
		if cleanErr := os.RemoveAll(dataDir); cleanErr != nil {
			klog.Error(log("failed to remove dir after error [%s]: %v", dataDir, cleanErr))
		}
		return err
	}
	defer func() {
		// Only if there was an error and volume operation was considered
		// finished, we should remove the directory.
		if err != nil && volumetypes.IsOperationFinishedError(err) {
			// clean up metadata
			klog.Errorf(log("attacher.MountDevice failed: %v", err))
			if err := removeMountDir(c.plugin, deviceMountPath); err != nil {
				klog.Error(log("attacher.MountDevice failed to remove mount dir after error [%s]: %v", deviceMountPath, err))
			}
		}
	}()

	if !stageUnstageSet {
		klog.Infof(log("attacher.MountDevice STAGE_UNSTAGE_VOLUME capability not set. Skipping MountDevice..."))
		// defer does *not* remove the metadata file and it's correct - UnmountDevice needs it there.
		return nil
	}

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := v1.ReadWriteOnce
	if spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = spec.PersistentVolume.Spec.AccessModes[0]
	}

	var mountOptions []string
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.MountOptions != nil {
		mountOptions = spec.PersistentVolume.Spec.MountOptions
	}

	fsType := csiSource.FSType
	err = csi.NodeStageVolume(ctx,
		csiSource.VolumeHandle,
		publishContext,
		deviceMountPath,
		fsType,
		accessMode,
		nodeStageSecrets,
		csiSource.VolumeAttributes,
		mountOptions)

	if err != nil {
		return err
	}

	klog.V(4).Infof(log("attacher.MountDevice successfully requested NodeStageVolume [%s]", deviceMountPath))
	return err
}

var _ volume.Detacher = &csiAttacher{}

var _ volume.DeviceUnmounter = &csiAttacher{}

func (c *csiAttacher) Detach(volumeName string, nodeName types.NodeName) error {
	var attachID string
	var volID string

	if volumeName == "" {
		klog.Error(log("detacher.Detach missing value for parameter volumeName"))
		return errors.New("missing expected parameter volumeName")
	}

	if isAttachmentName(volumeName) {
		// Detach can also be called with the attach ID as the `volumeName`. This codepath is
		// hit only when we have migrated an in-tree volume to CSI and the A/D Controller is shut
		// down, the pod with the volume is deleted, and the A/D Controller starts back up in that
		// order.
		attachID = volumeName

		// Vol ID should be the volume handle, except that is not available here.
		// It is only used in log messages so in the event that this happens log messages will be
		// printing out the attachID instead of the volume handle.
		volID = volumeName
	} else {
		// volumeName in format driverName<SEP>volumeHandle generated by plugin.GetVolumeName()
		parts := strings.Split(volumeName, volNameSep)
		if len(parts) != 2 {
			klog.Error(log("detacher.Detach insufficient info encoded in volumeName"))
			return errors.New("volumeName missing expected data")
		}

		driverName := parts[0]
		volID = parts[1]
		attachID = getAttachmentName(volID, driverName, string(nodeName))
	}

	if err := c.k8s.StorageV1().VolumeAttachments().Delete(context.TODO(), attachID, metav1.DeleteOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			// object deleted or never existed, done
			klog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] not found, object deleted", attachID, volID))
			return nil
		}
		return errors.New(log("detacher.Detach failed to delete VolumeAttachment [%s]: %v", attachID, err))
	}

	klog.V(4).Info(log("detacher deleted ok VolumeAttachment.ID=%s", attachID))
	err := c.waitForVolumeDetachment(volID, attachID, csiTimeout)
	return err
}

func (c *csiAttacher) waitForVolumeDetachment(volumeHandle, attachID string, timeout time.Duration) error {
	klog.V(4).Info(log("probing for updates from CSI driver for [attachment.ID=%v]", attachID))

	timer := time.NewTimer(timeout) // TODO (vladimirvivien) investigate making this configurable
	defer timer.Stop()

	return c.waitForVolumeDetachmentInternal(volumeHandle, attachID, timer, timeout)
}

func (c *csiAttacher) waitForVolumeDetachmentInternal(volumeHandle, attachID string, timer *time.Timer,
	timeout time.Duration) error {
	klog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))
	attach, err := c.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), attachID, meta.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			//object deleted or never existed, done
			klog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] not found, object deleted", attachID, volumeHandle))
			return nil
		}
		return errors.New(log("detacher.WaitForDetach failed for volume [%s] (will continue to try): %v", volumeHandle, err))
	}
	err = c.waitForVolumeAttachDetachStatus(attach, volumeHandle, attachID, timer, timeout, verifyDetachmentStatus)
	if err != nil {
		return err
	}
	return err
}

func (c *csiAttacher) waitForVolumeAttachDetachStatus(attach *storage.VolumeAttachment, volumeHandle, attachID string,
	timer *time.Timer, timeout time.Duration, verifyStatus verifyAttachDetachStatus) error {
	successful, err := verifyStatus(attach, volumeHandle)
	if err != nil {
		return err
	}
	if successful {
		return nil
	}

	watcher, err := c.k8s.StorageV1().VolumeAttachments().Watch(context.TODO(), meta.SingleObject(meta.ObjectMeta{Name: attachID, ResourceVersion: attach.ResourceVersion}))
	if err != nil {
		return fmt.Errorf("watch error:%v for volume %v", err, volumeHandle)
	}

	ch := watcher.ResultChan()
	defer watcher.Stop()
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				klog.Errorf("[attachment.ID=%v] watch channel had been closed", attachID)
				return errors.New("volume attachment watch channel had been closed")
			}

			switch event.Type {
			case watch.Added, watch.Modified:
				attach, _ := event.Object.(*storage.VolumeAttachment)
				successful, err := verifyStatus(attach, volumeHandle)
				if err != nil {
					return err
				}
				if successful {
					return nil
				}
			case watch.Deleted:
				// set attach nil to get different results
				// for detachment, a deleted event means successful detachment, should return success
				// for attachment, should return fail
				if successful, err := verifyStatus(nil, volumeHandle); !successful {
					return err
				}
				klog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] has been deleted", attachID, volumeHandle))
				return nil

			case watch.Error:
				klog.Warningf("waitForVolumeAttachDetachInternal received watch error: %v", event)
			}

		case <-timer.C:
			klog.Error(log("attachdetacher.WaitForDetach timeout after %v [volume=%v; attachment.ID=%v]", timeout, volumeHandle, attachID))
			return fmt.Errorf("attachdetachment timeout for volume %v", volumeHandle)
		}
	}
}

func (c *csiAttacher) UnmountDevice(deviceMountPath string) error {
	klog.V(4).Info(log("attacher.UnmountDevice(%s)", deviceMountPath))

	// Setup
	var driverName, volID string
	dataDir := filepath.Dir(deviceMountPath)
	data, err := loadVolumeData(dataDir, volDataFileName)
	if err == nil {
		driverName = data[volDataKey.driverName]
		volID = data[volDataKey.volHandle]
	} else {
		klog.Error(log("UnmountDevice failed to load volume data file [%s]: %v", dataDir, err))

		// The volume might have been mounted by old CSI volume plugin. Fall back to the old behavior: read PV from API server
		driverName, volID, err = getDriverAndVolNameFromDeviceMountPath(c.k8s, deviceMountPath)
		if err != nil {
			klog.Errorf(log("attacher.UnmountDevice failed to get driver and volume name from device mount path: %v", err))
			return err
		}
	}

	if c.csiClient == nil {
		c.csiClient, err = newCsiDriverClient(csiDriverName(driverName))
		if err != nil {
			return errors.New(log("attacher.UnmountDevice failed to create newCsiDriverClient: %v", err))
		}
	}
	csi := c.csiClient

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()
	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := csi.NodeSupportsStageUnstage(ctx)
	if err != nil {
		return errors.New(log("attacher.UnmountDevice failed to check whether STAGE_UNSTAGE_VOLUME set: %v", err))
	}
	if !stageUnstageSet {
		klog.Infof(log("attacher.UnmountDevice STAGE_UNSTAGE_VOLUME capability not set. Skipping UnmountDevice..."))
		// Just	delete the global directory + json file
		if err := removeMountDir(c.plugin, deviceMountPath); err != nil {
			return errors.New(log("failed to clean up global mount %s: %s", dataDir, err))
		}

		return nil
	}

	// Start UnmountDevice
	err = csi.NodeUnstageVolume(ctx,
		volID,
		deviceMountPath)

	if err != nil {
		return errors.New(log("attacher.UnmountDevice failed: %v", err))
	}

	// Delete the global directory + json file
	if err := removeMountDir(c.plugin, deviceMountPath); err != nil {
		return errors.New(log("failed to clean up global mount %s: %s", dataDir, err))
	}

	klog.V(4).Infof(log("attacher.UnmountDevice successfully requested NodeUnStageVolume [%s]", deviceMountPath))
	return nil
}

// getAttachmentName returns csi-<sha256(volName,csiDriverName,NodeName)>
func getAttachmentName(volName, csiDriverName, nodeName string) string {
	result := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", volName, csiDriverName, nodeName)))
	return fmt.Sprintf("csi-%x", result)
}

// isAttachmentName returns true if the string given is of the form of an Attach ID
// and false otherwise
func isAttachmentName(unknownString string) bool {
	// 68 == "csi-" + len(sha256hash)
	if strings.HasPrefix(unknownString, "csi-") && len(unknownString) == 68 {
		return true
	}
	return false
}

func makeDeviceMountPath(plugin *csiPlugin, spec *volume.Spec) (string, error) {
	if spec == nil {
		return "", errors.New(log("makeDeviceMountPath failed, spec is nil"))
	}

	pvName := spec.PersistentVolume.Name
	if pvName == "" {
		return "", errors.New(log("makeDeviceMountPath failed, pv name empty"))
	}

	return filepath.Join(plugin.host.GetPluginDir(plugin.GetPluginName()), persistentVolumeInGlobalPath, pvName, globalMountInGlobalPath), nil
}

func getDriverAndVolNameFromDeviceMountPath(k8s kubernetes.Interface, deviceMountPath string) (string, string, error) {
	// deviceMountPath structure: /var/lib/kubelet/plugins/kubernetes.io/csi/pv/{pvname}/globalmount
	dir := filepath.Dir(deviceMountPath)
	if file := filepath.Base(deviceMountPath); file != globalMountInGlobalPath {
		return "", "", errors.New(log("getDriverAndVolNameFromDeviceMountPath failed, path did not end in %s", globalMountInGlobalPath))
	}
	// dir is now /var/lib/kubelet/plugins/kubernetes.io/csi/pv/{pvname}
	pvName := filepath.Base(dir)

	// Get PV and check for errors
	pv, err := k8s.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, meta.GetOptions{})
	if err != nil {
		return "", "", err
	}
	if pv == nil || pv.Spec.CSI == nil {
		return "", "", errors.New(log("getDriverAndVolNameFromDeviceMountPath could not find CSI Persistent Volume Source for pv: %s", pvName))
	}

	// Get VolumeHandle and PluginName from pv
	csiSource := pv.Spec.CSI
	if csiSource.Driver == "" {
		return "", "", errors.New(log("getDriverAndVolNameFromDeviceMountPath failed, driver name empty"))
	}
	if csiSource.VolumeHandle == "" {
		return "", "", errors.New(log("getDriverAndVolNameFromDeviceMountPath failed, VolumeHandle empty"))
	}

	return csiSource.Driver, csiSource.VolumeHandle, nil
}

func verifyAttachmentStatus(attachment *storage.VolumeAttachment, volumeHandle string) (bool, error) {
	// when we received a deleted event during attachment, fail fast
	if attachment == nil {
		klog.Error(log("VolumeAttachment [%s] has been deleted, will not continue to wait for attachment", volumeHandle))
		return false, errors.New("volume attachment has been deleted")
	}
	// if being deleted, fail fast
	if attachment.GetDeletionTimestamp() != nil {
		klog.Error(log("VolumeAttachment [%s] has deletion timestamp, will not continue to wait for attachment", attachment.Name))
		return false, errors.New("volume attachment is being deleted")
	}
	// attachment OK
	if attachment.Status.Attached {
		return true, nil
	}
	// driver reports attach error
	attachErr := attachment.Status.AttachError
	if attachErr != nil {
		klog.Error(log("attachment for %v failed: %v", volumeHandle, attachErr.Message))
		return false, errors.New(attachErr.Message)
	}
	return false, nil
}

func verifyDetachmentStatus(attachment *storage.VolumeAttachment, volumeHandle string) (bool, error) {
	// when we received a deleted event during detachment
	// it means we have successfully detached it.
	if attachment == nil {
		return true, nil
	}
	// driver reports detach error
	detachErr := attachment.Status.DetachError
	if detachErr != nil {
		klog.Error(log("detachment for VolumeAttachment for volume [%s] failed: %v", volumeHandle, detachErr.Message))
		return false, errors.New(detachErr.Message)
	}
	return false, nil
}
