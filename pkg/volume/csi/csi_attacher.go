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

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/filesystem"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/utils/clock"
)

const globalMountInGlobalPath = "globalmount"

type csiAttacher struct {
	plugin       *csiPlugin
	k8s          kubernetes.Interface
	watchTimeout time.Duration

	csiClient csiClient
}

// volume.Attacher methods
var _ volume.Attacher = &csiAttacher{}

var _ volume.Detacher = &csiAttacher{}

var _ volume.DeviceMounter = &csiAttacher{}

func (c *csiAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	_, ok := c.plugin.host.(volume.KubeletVolumeHost)
	if ok {
		return "", errors.New("attaching volumes from the kubelet is not supported")
	}

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

	attachment, err := c.plugin.volumeAttachmentLister.Get(attachID)
	if err != nil && !apierrors.IsNotFound(err) {
		return "", errors.New(log("failed to get volume attachment from lister: %v", err))
	}

	if attachment == nil {
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
			ObjectMeta: metav1.ObjectMeta{
				Name: attachID,
			},
			Spec: storage.VolumeAttachmentSpec{
				NodeName: node,
				Attacher: pvSrc.Driver,
				Source:   vaSrc,
			},
		}

		_, err = c.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
		if err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return "", errors.New(log("attacher.Attach failed: %v", err))
			}
			klog.V(4).Info(log("attachment [%v] for volume [%v] already exists (will not be recreated)", attachID, pvSrc.VolumeHandle))
		} else {
			klog.V(4).Info(log("attachment [%v] for volume [%v] created successfully", attachID, pvSrc.VolumeHandle))
		}
	}

	// Attach and detach functionality is exclusive to the CSI plugin that runs in the AttachDetachController,
	// and has access to a VolumeAttachment lister that can be polled for the current status.
	if err := c.waitForVolumeAttachmentWithLister(spec, pvSrc.VolumeHandle, attachID, c.watchTimeout); err != nil {
		return "", err
	}

	klog.V(4).Info(log("attacher.Attach finished OK with VolumeAttachment object [%s]", attachID))

	// Don't return attachID as a devicePath. We can reconstruct the attachID using getAttachmentName()
	return "", nil
}

// WaitForAttach waits for the attach operation to complete and returns the device path when it is done.
// But in this case, there should be no waiting. The device is found by the CSI driver later, in NodeStage / NodePublish calls.
// so it should just return device metadata, in this case it is VolumeAttachment name. If the target VolumeAttachment does not
// exist or is not attached, the function will return an error. And then the caller (kubelet) should retry it.
// We can get rid of watching it that serves no purpose. More details in https://issues.k8s.io/124398
func (c *csiAttacher) WaitForAttach(spec *volume.Spec, _ string, pod *v1.Pod, _ time.Duration) (string, error) {
	source, err := getPVSourceFromSpec(spec)
	if err != nil {
		return "", errors.New(log("attacher.WaitForAttach failed to extract CSI volume source: %v", err))
	}

	volumeHandle := source.VolumeHandle
	attachID := getAttachmentName(source.VolumeHandle, source.Driver, string(c.plugin.host.GetNodeName()))

	attach, err := c.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), attachID, metav1.GetOptions{})
	if err != nil {
		klog.Error(log("attacher.WaitForAttach failed for volume [%s] (will continue to try): %v", volumeHandle, err))
		return "", fmt.Errorf("volume %v has GET error for volume attachment %v: %v", volumeHandle, attachID, err)
	}

	successful, err := verifyAttachmentStatus(attach, volumeHandle)
	if err != nil {
		return "", err
	}
	if !successful {
		klog.Error(log("attacher.WaitForAttach failed for volume [%s] attached (will continue to try)", volumeHandle))
		return "", fmt.Errorf("volume %v is not attached for volume attachment %v", volumeHandle, attachID)
	}
	return attach.Name, nil
}

func (c *csiAttacher) waitForVolumeAttachmentWithLister(spec *volume.Spec, volumeHandle, attachID string, timeout time.Duration) error {
	klog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))

	verifyStatus := func() (bool, error) {
		volumeAttachment, err := c.plugin.volumeAttachmentLister.Get(attachID)
		if err != nil {
			// Ignore "not found" errors in case the VolumeAttachment was just created and hasn't yet made it into the lister.
			if !apierrors.IsNotFound(err) {
				klog.Error(log("unexpected error waiting for volume attachment, %v", err))
				return false, err
			}

			// The VolumeAttachment is not available yet and we will have to try again.
			return false, nil
		}

		successful, err := verifyAttachmentStatus(volumeAttachment, volumeHandle)
		if err != nil {
			return false, err
		}
		return successful, nil
	}

	return c.waitForVolumeAttachDetachStatusWithLister(spec, volumeHandle, attachID, timeout, verifyStatus, "Attach")
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
		attach, err = c.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), attachID, metav1.GetOptions{})
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

func (c *csiAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string, deviceMounterArgs volume.DeviceMounterArgs) error {
	klog.V(4).Infof(log("attacher.MountDevice(%s, %s)", devicePath, deviceMountPath))

	if deviceMountPath == "" {
		return errors.New(log("attacher.MountDevice failed, deviceMountPath is empty"))
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
			// Treat the absence of the CSI driver as a transient error
			// See https://github.com/kubernetes/kubernetes/issues/120268
			return volumetypes.NewTransientOperationFailure(log("attacher.MountDevice failed to create newCsiDriverClient: %v", err))
		}
	}
	csi := c.csiClient

	ctx, cancel := createCSIOperationContext(spec, c.watchTimeout)
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

	var mountOptions []string
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.MountOptions != nil {
		mountOptions = spec.PersistentVolume.Spec.MountOptions
	}

	var seLinuxSupported bool
	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
		support, err := c.plugin.SupportsSELinuxContextMount(spec)
		if err != nil {
			return errors.New(log("failed to query for SELinuxMount support: %s", err))
		}
		if support && deviceMounterArgs.SELinuxLabel != "" {
			mountOptions = util.AddSELinuxMountOption(mountOptions, deviceMounterArgs.SELinuxLabel)
			seLinuxSupported = true
		}
	}

	// Store volume metadata for UnmountDevice. Keep it around even if the
	// driver does not support NodeStage, UnmountDevice still needs it.
	if err = filesystem.MkdirAllWithPathCheck(deviceMountPath, 0750); err != nil {
		return errors.New(log("attacher.MountDevice failed to create dir %#v:  %v", deviceMountPath, err))
	}

	klog.V(4).Info(log("created target path successfully [%s]", deviceMountPath))
	dataDir := filepath.Dir(deviceMountPath)
	data := map[string]string{
		volDataKey.volHandle:  csiSource.VolumeHandle,
		volDataKey.driverName: csiSource.Driver,
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) && seLinuxSupported {
		data[volDataKey.seLinuxMountContext] = deviceMounterArgs.SELinuxLabel
	}

	err = saveVolumeData(dataDir, volDataFileName, data)
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

	if err != nil {
		errMsg := log("failed to save volume info data: %v", err)
		klog.Error(errMsg)
		return errors.New(errMsg)
	}

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

	var nodeStageFSGroupArg *int64
	driverSupportsCSIVolumeMountGroup, err := csi.NodeSupportsVolumeMountGroup(ctx)
	if err != nil {
		return volumetypes.NewTransientOperationFailure(log("attacher.MountDevice failed to determine if the node service has VOLUME_MOUNT_GROUP capability: %v", err))
	}

	if driverSupportsCSIVolumeMountGroup {
		klog.V(3).Infof("Driver %s supports applying FSGroup (has VOLUME_MOUNT_GROUP node capability). Delegating FSGroup application to the driver through NodeStageVolume.", csiSource.Driver)
		nodeStageFSGroupArg = deviceMounterArgs.FsGroup
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
		mountOptions,
		nodeStageFSGroupArg)

	if err != nil {
		return err
	}

	klog.V(4).Infof(log("attacher.MountDevice successfully requested NodeStageVolume [%s]", deviceMountPath))
	return err
}

var _ volume.Detacher = &csiAttacher{}

var _ volume.DeviceUnmounter = &csiAttacher{}

func (c *csiAttacher) Detach(volumeName string, nodeName types.NodeName) error {
	_, ok := c.plugin.host.(volume.KubeletVolumeHost)
	if ok {
		return errors.New("detaching volumes from the kubelet is not supported")
	}

	var attachID string
	var volID string

	if volumeName == "" {
		klog.Error(log("detacher.Detach missing value for parameter volumeName"))
		return errors.New("missing expected parameter volumeName")
	}

	// volumeName in format driverName<SEP>volumeHandle generated by plugin.GetVolumeName()
	parts := strings.Split(volumeName, volNameSep)
	if len(parts) != 2 {
		klog.Error(log("detacher.Detach insufficient info encoded in volumeName"))
		return errors.New("volumeName missing expected data")
	}

	driverName := parts[0]
	volID = parts[1]
	attachID = getAttachmentName(volID, driverName, string(nodeName))

	if err := c.k8s.StorageV1().VolumeAttachments().Delete(context.TODO(), attachID, metav1.DeleteOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			// object deleted or never existed, done
			klog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] not found, object deleted", attachID, volID))
			return nil
		}
		return errors.New(log("detacher.Detach failed to delete VolumeAttachment [%s]: %v", attachID, err))
	}

	klog.V(4).Info(log("detacher deleted ok VolumeAttachment.ID=%s", attachID))

	// Attach and detach functionality is exclusive to the CSI plugin that runs in the AttachDetachController,
	// and has access to a VolumeAttachment lister that can be polled for the current status.
	return c.waitForVolumeDetachmentWithLister(volID, attachID, c.watchTimeout)
}

func (c *csiAttacher) waitForVolumeDetachmentWithLister(volumeHandle, attachID string, timeout time.Duration) error {
	klog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))

	verifyStatus := func() (bool, error) {
		volumeAttachment, err := c.plugin.volumeAttachmentLister.Get(attachID)
		if err != nil {
			if !apierrors.IsNotFound(err) {
				return false, errors.New(log("detacher.WaitForDetach failed for volume [%s] (will continue to try): %v", volumeHandle, err))
			}

			// Detachment successful.
			klog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] not found, object deleted", attachID, volumeHandle))
			return true, nil
		}

		// Detachment is only "successful" once the VolumeAttachment is deleted, however we perform
		// this check to make sure the object does not contain any detach errors.
		successful, err := verifyDetachmentStatus(volumeAttachment, volumeHandle)
		if err != nil {
			return false, err
		}
		return successful, nil
	}

	return c.waitForVolumeAttachDetachStatusWithLister(nil, volumeHandle, attachID, timeout, verifyStatus, "Detach")
}

func (c *csiAttacher) waitForVolumeAttachDetachStatusWithLister(spec *volume.Spec, volumeHandle, attachID string, timeout time.Duration, verifyStatus func() (bool, error), operation string) error {
	var (
		initBackoff = 500 * time.Millisecond
		// This is approximately the duration between consecutive ticks after two minutes (CSI timeout).
		maxBackoff    = 7 * time.Second
		resetDuration = time.Minute
		backoffFactor = 1.05
		jitter        = 0.1
		clock         = &clock.RealClock{}
	)
	backoffMgr := wait.NewExponentialBackoffManager(initBackoff, maxBackoff, resetDuration, backoffFactor, jitter, clock)

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Get driver name from spec for better log messages. During detach spec can be nil, and it's ok for driver to be unknown.
	csiDriverName, err := GetCSIDriverName(spec)
	if err != nil {
		csiDriverName = "unknown"
		klog.V(4).Info(log("Could not find CSI driver name in spec for volume [%v]", volumeHandle))
	}

	for {
		t := backoffMgr.Backoff()
		select {
		case <-t.C():
			successful, err := verifyStatus()
			if err != nil {
				return err
			}
			if successful {
				return nil
			}
		case <-ctx.Done():
			t.Stop()
			klog.Error(log("%s timeout after %v [volume=%v; attachment.ID=%v]", operation, timeout, volumeHandle, attachID))
			return fmt.Errorf("timed out waiting for external-attacher of %v CSI driver to %v volume %v", csiDriverName, strings.ToLower(operation), volumeHandle)
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
		if errors.Is(err, os.ErrNotExist) {
			klog.V(4).Info(log("attacher.UnmountDevice skipped because volume data file [%s] does not exist", dataDir))
			return nil
		}

		klog.Errorf(log("attacher.UnmountDevice failed to get driver and volume name from device mount path: %v", err))
		return err
	}

	if c.csiClient == nil {
		c.csiClient, err = newCsiDriverClient(csiDriverName(driverName))
		if err != nil {
			// Treat the absence of the CSI driver as a transient error
			// See https://github.com/kubernetes/kubernetes/issues/120268
			return volumetypes.NewTransientOperationFailure(log("attacher.UnmountDevice failed to create newCsiDriverClient: %v", err))
		}
	}
	csi := c.csiClient

	// could not get whether this is migrated because there is no spec
	ctx, cancel := createCSIOperationContext(nil, csiTimeout)
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

func makeDeviceMountPath(plugin *csiPlugin, spec *volume.Spec) (string, error) {
	if spec == nil {
		return "", errors.New("makeDeviceMountPath failed, spec is nil")
	}

	driver, err := GetCSIDriverName(spec)
	if err != nil {
		return "", err
	}
	if driver == "" {
		return "", errors.New("makeDeviceMountPath failed, csi source driver name is empty")
	}

	csiSource, err := getPVSourceFromSpec(spec)
	if err != nil {
		return "", errors.New(log("makeDeviceMountPath failed to get CSIPersistentVolumeSource: %v", err))
	}

	if csiSource.VolumeHandle == "" {
		return "", errors.New("makeDeviceMountPath failed, CSIPersistentVolumeSource volume handle is empty")
	}

	result := sha256.Sum256([]byte(fmt.Sprintf("%s", csiSource.VolumeHandle)))
	volSha := fmt.Sprintf("%x", result)
	return filepath.Join(plugin.host.GetPluginDir(plugin.GetPluginName()), driver, volSha, globalMountInGlobalPath), nil
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
