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
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/golang/glog"
	grpctx "golang.org/x/net/context"

	csipb "github.com/container-storage-interface/spec/lib/go/csi/v0"
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/volume"
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
	// the primary intent of the enclosing method is to create VolumeAttachment.
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

	timer := time.NewTimer(timeout) // TODO (vladimirvivien) investigate making this configurable
	defer timer.Stop()

	return c.waitForVolumeAttachmentInternal(volumeHandle, attachID, timer, timeout)
}

func (c *csiAttacher) waitForVolumeAttachmentInternal(volumeHandle, attachID string, timer *time.Timer, timeout time.Duration) (string, error) {
	glog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))
	attach, err := c.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		glog.Error(log("attacher.WaitForAttach failed for volume [%s] (will continue to try): %v", volumeHandle, err))
		return "", err
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

	watcher, err := c.k8s.StorageV1beta1().VolumeAttachments().Watch(meta.SingleObject(meta.ObjectMeta{Name: attachID, ResourceVersion: attach.ResourceVersion}))
	if err != nil {
		return "", fmt.Errorf("watch error:%v for volume %v", err, volumeHandle)
	}

	ch := watcher.ResultChan()
	defer watcher.Stop()

	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.Errorf("[attachment.ID=%v] watch channel had been closed", attachID)
				return "", errors.New("volume attachment watch channel had been closed")
			}

			switch event.Type {
			case watch.Added, watch.Modified:
				attach, _ := event.Object.(*storage.VolumeAttachment)
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
			case watch.Deleted:
				// if deleted, fail fast
				glog.Error(log("VolumeAttachment [%s] has been deleted, will not continue to wait for attachment", attachID))
				return "", errors.New("volume attachment has been deleted")

			case watch.Error:
				// start another cycle
				c.waitForVolumeAttachmentInternal(volumeHandle, attachID, timer, timeout)
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
	glog.V(4).Info(log("attacher.GetDeviceMountPath(%v)", spec))
	deviceMountPath, err := makeDeviceMountPath(c.plugin, spec)
	if err != nil {
		glog.Error(log("attacher.GetDeviceMountPath failed to make device mount path: %v", err))
		return "", err
	}
	glog.V(4).Infof("attacher.GetDeviceMountPath succeeded, deviceMountPath: %s", deviceMountPath)
	return deviceMountPath, nil
}

func (c *csiAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	glog.V(4).Infof(log("attacher.MountDevice(%s, %s)", devicePath, deviceMountPath))

	mounted, err := isDirMounted(c.plugin, deviceMountPath)
	if err != nil {
		glog.Error(log("attacher.MountDevice failed while checking mount status for dir [%s]", deviceMountPath))
		return err
	}

	if mounted {
		glog.V(4).Info(log("attacher.MountDevice skipping mount, dir already mounted [%s]", deviceMountPath))
		return nil
	}

	// Setup
	if spec == nil {
		return fmt.Errorf("attacher.MountDevice failed, spec is nil")
	}
	csiSource, err := getCSISourceFromSpec(spec)
	if err != nil {
		glog.Error(log("attacher.MountDevice failed to get CSI persistent source: %v", err))
		return err
	}

	if c.csiClient == nil {
		if csiSource.Driver == "" {
			return fmt.Errorf("attacher.MountDevice failed, driver name is empty")
		}
		addr := fmt.Sprintf(csiAddrTemplate, csiSource.Driver)
		c.csiClient = newCsiDriverClient("unix", addr)
	}
	csi := c.csiClient

	ctx, cancel := grpctx.WithTimeout(grpctx.Background(), csiTimeout)
	defer cancel()
	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := hasStageUnstageCapability(ctx, csi)
	if err != nil {
		glog.Error(log("attacher.MountDevice failed to check STAGE_UNSTAGE_VOLUME: %v", err))
		return err
	}
	if !stageUnstageSet {
		glog.Infof(log("attacher.MountDevice STAGE_UNSTAGE_VOLUME capability not set. Skipping MountDevice..."))
		return nil
	}

	// Start MountDevice
	if deviceMountPath == "" {
		return fmt.Errorf("attacher.MountDevice failed, deviceMountPath is empty")
	}

	nodeName := string(c.plugin.host.GetNodeName())
	attachID := getAttachmentName(csiSource.VolumeHandle, csiSource.Driver, nodeName)

	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	attachment, err := c.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		glog.Error(log("attacher.MountDevice failed while getting volume attachment [id=%v]: %v", attachID, err))
		return err
	}

	if attachment == nil {
		glog.Error(log("unable to find VolumeAttachment [id=%s]", attachID))
		return errors.New("no existing VolumeAttachment found")
	}
	publishVolumeInfo := attachment.Status.AttachmentMetadata

	nodeStageSecrets := map[string]string{}
	if csiSource.NodeStageSecretRef != nil {
		nodeStageSecrets, err = getCredentialsFromSecret(c.k8s, csiSource.NodeStageSecretRef)
		if err != nil {
			return fmt.Errorf("fetching NodeStageSecretRef %s/%s failed: %v",
				csiSource.NodeStageSecretRef.Namespace, csiSource.NodeStageSecretRef.Name, err)
		}
	}

	// create target_dir before call to NodeStageVolume
	if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
		glog.Error(log("attacher.MountDevice failed to create dir %#v:  %v", deviceMountPath, err))
		return err
	}
	glog.V(4).Info(log("created target path successfully [%s]", deviceMountPath))

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := v1.ReadWriteOnce
	if spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = spec.PersistentVolume.Spec.AccessModes[0]
	}

	fsType := csiSource.FSType
	if len(fsType) == 0 {
		fsType = defaultFSType
	}

	err = csi.NodeStageVolume(ctx,
		csiSource.VolumeHandle,
		publishVolumeInfo,
		deviceMountPath,
		fsType,
		accessMode,
		nodeStageSecrets,
		csiSource.VolumeAttributes)

	if err != nil {
		glog.Errorf(log("attacher.MountDevice failed: %v", err))
		if err := removeMountDir(c.plugin, deviceMountPath); err != nil {
			glog.Error(log("attacher.MountDevice failed to remove mount dir after a NodeStageVolume() error [%s]: %v", deviceMountPath, err))
			return err
		}
		return err
	}

	glog.V(4).Infof(log("attacher.MountDevice successfully requested NodeStageVolume [%s]", deviceMountPath))
	return nil
}

var _ volume.Detacher = &csiAttacher{}

func (c *csiAttacher) Detach(volumeName string, nodeName types.NodeName) error {
	// volumeName in format driverName<SEP>volumeHandle generated by plugin.GetVolumeName()
	if volumeName == "" {
		glog.Error(log("detacher.Detach missing value for parameter volumeName"))
		return errors.New("missing expected parameter volumeName")
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

	timeout := c.waitSleepTime * 10
	timer := time.NewTimer(timeout) // TODO (vladimirvivien) investigate making this configurable
	defer timer.Stop()

	return c.waitForVolumeDetachmentInternal(volumeHandle, attachID, timer, timeout)
}

func (c *csiAttacher) waitForVolumeDetachmentInternal(volumeHandle, attachID string, timer *time.Timer, timeout time.Duration) error {
	glog.V(4).Info(log("probing VolumeAttachment [id=%v]", attachID))
	attach, err := c.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		if apierrs.IsNotFound(err) {
			//object deleted or never existed, done
			glog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] not found, object deleted", attachID, volumeHandle))
			return nil
		}
		glog.Error(log("detacher.WaitForDetach failed for volume [%s] (will continue to try): %v", volumeHandle, err))
		return err
	}
	// driver reports attach error
	detachErr := attach.Status.DetachError
	if detachErr != nil {
		glog.Error(log("detachment for VolumeAttachment [%v] for volume [%s] failed: %v", attachID, volumeHandle, detachErr.Message))
		return errors.New(detachErr.Message)
	}

	watcher, err := c.k8s.StorageV1beta1().VolumeAttachments().Watch(meta.SingleObject(meta.ObjectMeta{Name: attachID, ResourceVersion: attach.ResourceVersion}))
	if err != nil {
		return fmt.Errorf("watch error:%v for volume %v", err, volumeHandle)
	}
	ch := watcher.ResultChan()
	defer watcher.Stop()

	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.Errorf("[attachment.ID=%v] watch channel had been closed", attachID)
				return errors.New("volume attachment watch channel had been closed")
			}

			switch event.Type {
			case watch.Added, watch.Modified:
				attach, _ := event.Object.(*storage.VolumeAttachment)
				// driver reports attach error
				detachErr := attach.Status.DetachError
				if detachErr != nil {
					glog.Error(log("detachment for VolumeAttachment [%v] for volume [%s] failed: %v", attachID, volumeHandle, detachErr.Message))
					return errors.New(detachErr.Message)
				}
			case watch.Deleted:
				//object deleted
				glog.V(4).Info(log("VolumeAttachment object [%v] for volume [%v] has been deleted", attachID, volumeHandle))
				return nil

			case watch.Error:
				// start another cycle
				c.waitForVolumeDetachmentInternal(volumeHandle, attachID, timer, timeout)
			}

		case <-timer.C:
			glog.Error(log("detacher.WaitForDetach timeout after %v [volume=%v; attachment.ID=%v]", timeout, volumeHandle, attachID))
			return fmt.Errorf("detachment timeout for volume %v", volumeHandle)
		}
	}
}

func (c *csiAttacher) UnmountDevice(deviceMountPath string) error {
	glog.V(4).Info(log("attacher.UnmountDevice(%s)", deviceMountPath))

	// Setup
	driverName, volID, err := getDriverAndVolNameFromDeviceMountPath(c.k8s, deviceMountPath)
	if err != nil {
		glog.Errorf(log("attacher.UnmountDevice failed to get driver and volume name from device mount path: %v", err))
		return err
	}

	if c.csiClient == nil {
		addr := fmt.Sprintf(csiAddrTemplate, driverName)
		c.csiClient = newCsiDriverClient("unix", addr)
	}
	csi := c.csiClient

	ctx, cancel := grpctx.WithTimeout(grpctx.Background(), csiTimeout)
	defer cancel()
	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := hasStageUnstageCapability(ctx, csi)
	if err != nil {
		glog.Errorf(log("attacher.UnmountDevice failed to check whether STAGE_UNSTAGE_VOLUME set: %v", err))
		return err
	}
	if !stageUnstageSet {
		glog.Infof(log("attacher.UnmountDevice STAGE_UNSTAGE_VOLUME capability not set. Skipping UnmountDevice..."))
		return nil
	}

	// Start UnmountDevice
	err = csi.NodeUnstageVolume(ctx,
		volID,
		deviceMountPath)

	if err != nil {
		glog.Errorf(log("attacher.UnmountDevice failed: %v", err))
		return err
	}

	glog.V(4).Infof(log("attacher.UnmountDevice successfully requested NodeStageVolume [%s]", deviceMountPath))
	return nil
}

func hasStageUnstageCapability(ctx grpctx.Context, csi csiClient) (bool, error) {
	capabilities, err := csi.NodeGetCapabilities(ctx)
	if err != nil {
		return false, err
	}

	stageUnstageSet := false
	if capabilities == nil {
		return false, nil
	}
	for _, capability := range capabilities {
		if capability.GetRpc().GetType() == csipb.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME {
			stageUnstageSet = true
		}
	}
	return stageUnstageSet, nil
}

// getAttachmentName returns csi-<sha252(volName,csiDriverName,NodeName>
func getAttachmentName(volName, csiDriverName, nodeName string) string {
	result := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", volName, csiDriverName, nodeName)))
	return fmt.Sprintf("csi-%x", result)
}

func makeDeviceMountPath(plugin *csiPlugin, spec *volume.Spec) (string, error) {
	if spec == nil {
		return "", fmt.Errorf("makeDeviceMountPath failed, spec is nil")
	}

	pvName := spec.PersistentVolume.Name
	if pvName == "" {
		return "", fmt.Errorf("makeDeviceMountPath failed, pv name empty")
	}

	return path.Join(plugin.host.GetPluginDir(plugin.GetPluginName()), persistentVolumeInGlobalPath, pvName, globalMountInGlobalPath), nil
}

func getDriverAndVolNameFromDeviceMountPath(k8s kubernetes.Interface, deviceMountPath string) (string, string, error) {
	// deviceMountPath structure: /var/lib/kubelet/plugins/kubernetes.io/csi/pv/{pvname}/globalmount
	dir := filepath.Dir(deviceMountPath)
	if file := filepath.Base(deviceMountPath); file != globalMountInGlobalPath {
		return "", "", fmt.Errorf("getDriverAndVolNameFromDeviceMountPath failed, path did not end in %s", globalMountInGlobalPath)
	}
	// dir is now /var/lib/kubelet/plugins/kubernetes.io/csi/pv/{pvname}
	pvName := filepath.Base(dir)

	// Get PV and check for errors
	pv, err := k8s.CoreV1().PersistentVolumes().Get(pvName, meta.GetOptions{})
	if err != nil {
		return "", "", err
	}
	if pv == nil || pv.Spec.CSI == nil {
		return "", "", fmt.Errorf("getDriverAndVolNameFromDeviceMountPath could not find CSI Persistent Volume Source for pv: %s", pvName)
	}

	// Get VolumeHandle and PluginName from pv
	csiSource := pv.Spec.CSI
	if csiSource.Driver == "" {
		return "", "", fmt.Errorf("getDriverAndVolNameFromDeviceMountPath failed, driver name empty")
	}
	if csiSource.VolumeHandle == "" {
		return "", "", fmt.Errorf("getDriverAndVolNameFromDeviceMountPath failed, VolumeHandle empty")
	}

	return csiSource.Driver, csiSource.VolumeHandle, nil
}
