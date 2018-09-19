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
	"os"
	"path"
	"path/filepath"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1beta1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	ioutil "k8s.io/kubernetes/pkg/volume/util"
)

type csiBlockMapper struct {
	k8s        kubernetes.Interface
	csiClient  csiClient
	plugin     *csiPlugin
	driverName string
	specName   string
	volumeID   string
	readOnly   bool
	spec       *volume.Spec
	podUID     types.UID
	volumeInfo map[string]string
}

var _ volume.BlockVolumeMapper = &csiBlockMapper{}

// GetGlobalMapPath returns a global map path (on the node) to a device file which will be symlinked to
// Example: plugins/kubernetes.io/csi/volumeDevices/{volumeID}/dev
func (m *csiBlockMapper) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	dir := getVolumeDevicePluginDir(spec.Name(), m.plugin.host)
	klog.V(4).Infof(log("blockMapper.GetGlobalMapPath = %s", dir))
	return dir, nil
}

// getStagingPath returns a staging path (on the node) to a device file which will be bind mounted to
// Example: plugins/kubernetes.io/csi/volumeDevices/staging/{volumeID}
func (m *csiBlockMapper) getStagingPath() string {
	sanitizedSpecVolID := kstrings.EscapeQualifiedNameForDisk(m.specName)
	return path.Join(m.plugin.host.GetVolumeDevicePluginDir(csiPluginName), "staging", sanitizedSpecVolID)
}

// getPublishPath returns a publish path (on the node) to a device file which will be bind mounted to
// Example: plugins/kubernetes.io/csi/volumeDevices/publish/{volumeID}
func (m *csiBlockMapper) getPublishPath() string {
	sanitizedSpecVolID := kstrings.EscapeQualifiedNameForDisk(m.specName)
	return path.Join(m.plugin.host.GetVolumeDevicePluginDir(csiPluginName), "publish", sanitizedSpecVolID)
}

// GetPodDeviceMapPath returns pod's device file which will be mapped to a volume
// returns: pods/{podUid}/volumeDevices/kubernetes.io~csi, {volumeID}
func (m *csiBlockMapper) GetPodDeviceMapPath() (string, string) {
	path := m.plugin.host.GetPodVolumeDeviceDir(m.podUID, kstrings.EscapeQualifiedNameForDisk(csiPluginName))
	specName := m.specName
	klog.V(4).Infof(log("blockMapper.GetPodDeviceMapPath [path=%s; name=%s]", path, specName))
	return path, specName
}

// stageVolumeForBlock makes a block volume visible via the staging path
func (m *csiBlockMapper) stageVolumeForBlock(
	ctx context.Context,
	csi csiClient,
	accessMode v1.PersistentVolumeAccessMode,
	csiSource *v1.CSIPersistentVolumeSource,
	attachment *storage.VolumeAttachment,
) (string, error) {
	klog.V(4).Infof(log("blockMapper.stageVolumeForBlock called"))

	stagingPath := m.getStagingPath()
	klog.V(4).Infof(log("blockMapper.stageVolumeForBlock stagingPath set [%s]", stagingPath))

	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := hasStageUnstageCapability(ctx, csi)
	if err != nil {
		klog.Error(log("blockMapper.stageVolumeForBlock failed to check STAGE_UNSTAGE_VOLUME capability: %v", err))
		return "", err
	}
	if !stageUnstageSet {
		klog.Infof(log("blockMapper.stageVolumeForBlock STAGE_UNSTAGE_VOLUME capability not set. Skipping MountDevice..."))
		return "", nil
	}

	// Start MountDevice
	publishVolumeInfo := attachment.Status.AttachmentMetadata

	nodeStageSecrets := map[string]string{}
	if csiSource.NodeStageSecretRef != nil {
		nodeStageSecrets, err = getCredentialsFromSecret(m.k8s, csiSource.NodeStageSecretRef)
		if err != nil {
			return "", fmt.Errorf("failed to get NodeStageSecretRef %s/%s: %v",
				csiSource.NodeStageSecretRef.Namespace, csiSource.NodeStageSecretRef.Name, err)
		}
	}

	// setup path directory for stagingPath before call to NodeStageVolume
	stagingDir := filepath.Dir(stagingPath)
	if err := os.MkdirAll(stagingDir, 0750); err != nil {
		klog.Error(log("blockMapper.stageVolumeForBlock failed to create dir %s: %v", stagingDir, err))
		return "", err
	}
	klog.V(4).Info(log("blockMapper.stageVolumeForBlock created directory for stagingPath successfully [%s]", stagingDir))

	// create an empty file on staging path where block device is bind mounted to
	stagingPathFile, err := os.OpenFile(stagingPath, os.O_CREATE|os.O_RDWR, 0750)
	if err != nil {
		klog.Error(log("blockMapper.stageVolumeForBlock failed to create file %s: %v", stagingPathFile, err))
		return "", err
	}
	if err := stagingPathFile.Close(); err != nil {
		klog.Error(log("blockMapper.stageVolumeForBlock failed to close file %s: %v", stagingPathFile, err))
		return "", err
	}
	klog.V(4).Info(log("blockMapper.stageVolumeForBlock created an empty file on staging path successfully [%s]", stagingPathFile))

	// Request to attach the device to the node and to bind mount the device to stagingPath.
	err = csi.NodeStageVolume(ctx,
		csiSource.VolumeHandle,
		publishVolumeInfo,
		stagingPath,
		fsTypeBlockName,
		accessMode,
		nodeStageSecrets,
		csiSource.VolumeAttributes)

	if err != nil {
		klog.Error(log("blockMapper.stageVolumeForBlock failed: %v", err))
		return "", err
	}

	klog.V(4).Infof(log("blockMapper.stageVolumeForBlock successfully requested NodeStageVolume [%s]", stagingPath))
	return stagingPath, nil
}

// publishVolumeForBlock makes a block volume visible via the publish path
func (m *csiBlockMapper) publishVolumeForBlock(
	ctx context.Context,
	csi csiClient,
	accessMode v1.PersistentVolumeAccessMode,
	csiSource *v1.CSIPersistentVolumeSource,
	attachment *storage.VolumeAttachment,
	stagingPath string,
) (string, error) {
	klog.V(4).Infof(log("blockMapper.publishVolumeForBlock called"))

	publishVolumeInfo := attachment.Status.AttachmentMetadata

	nodePublishSecrets := map[string]string{}
	var err error
	if csiSource.NodePublishSecretRef != nil {
		nodePublishSecrets, err = getCredentialsFromSecret(m.k8s, csiSource.NodePublishSecretRef)
		if err != nil {
			klog.Errorf("blockMapper.publishVolumeForBlock failed to get NodePublishSecretRef %s/%s: %v",
				csiSource.NodePublishSecretRef.Namespace, csiSource.NodePublishSecretRef.Name, err)
			return "", err
		}
	}

	publishPath := m.getPublishPath()
	// setup path directory for stagingPath before call to NodeStageVolume
	publishDir := filepath.Dir(publishPath)
	if err := os.MkdirAll(publishDir, 0750); err != nil {
		klog.Error(log("blockMapper.publishVolumeForBlock failed to create dir %s:  %v", publishDir, err))
		return "", err
	}
	klog.V(4).Info(log("blockMapper.publishVolumeForBlock created directory for publishPath successfully [%s]", publishDir))

	// create an empty file on publish path where block device is bind mounted to
	publishPathFile, err := os.OpenFile(publishPath, os.O_CREATE|os.O_RDWR, 0750)
	if err != nil {
		klog.Error(log("blockMapper.publishVolumeForBlock failed to create file %s: %v", publishPathFile, err))
		return "", err
	}
	if err := publishPathFile.Close(); err != nil {
		klog.Error(log("blockMapper.publishVolumeForBlock failed to close file %s: %v", publishPathFile, err))
		return "", err
	}
	klog.V(4).Info(log("blockMapper.publishVolumeForBlock created an empty file on publish path successfully [%s]", publishPathFile))

	// Request to bind mount the device to publishPath.
	// If driver doesn't implement NodeStageVolume, attaching the device to the node is required, here.
	err = csi.NodePublishVolume(
		ctx,
		m.volumeID,
		m.readOnly,
		stagingPath,
		publishPath,
		accessMode,
		publishVolumeInfo,
		csiSource.VolumeAttributes,
		nodePublishSecrets,
		fsTypeBlockName,
		[]string{},
	)

	if err != nil {
		klog.Errorf(log("blockMapper.publishVolumeForBlock failed: %v", err))
		return "", err
	}

	return publishPath, nil
}

// SetUpDevice ensures the device is attached returns path where the device is located.
func (m *csiBlockMapper) SetUpDevice() (string, error) {
	if !m.plugin.blockEnabled {
		return "", errors.New("CSIBlockVolume feature not enabled")
	}
	klog.V(4).Infof(log("blockMapper.SetUpDevice called"))

	// Get csiSource from spec
	if m.spec == nil {
		klog.Error(log("blockMapper.SetUpDevice spec is nil"))
		return "", fmt.Errorf("spec is nil")
	}

	csiSource, err := getCSISourceFromSpec(m.spec)
	if err != nil {
		klog.Error(log("blockMapper.SetUpDevice failed to get CSI persistent source: %v", err))
		return "", err
	}

	// Search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	nodeName := string(m.plugin.host.GetNodeName())
	attachID := getAttachmentName(csiSource.VolumeHandle, csiSource.Driver, nodeName)
	attachment, err := m.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		klog.Error(log("blockMapper.SetupDevice failed to get volume attachment [id=%v]: %v", attachID, err))
		return "", err
	}

	if attachment == nil {
		klog.Error(log("blockMapper.SetupDevice unable to find VolumeAttachment [id=%s]", attachID))
		return "", errors.New("no existing VolumeAttachment found")
	}

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := v1.ReadWriteOnce
	if m.spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = m.spec.PersistentVolume.Spec.AccessModes[0]
	}

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	// Call NodeStageVolume
	stagingPath, err := m.stageVolumeForBlock(ctx, m.csiClient, accessMode, csiSource, attachment)
	if err != nil {
		return "", err
	}

	// Call NodePublishVolume
	publishPath, err := m.publishVolumeForBlock(ctx, m.csiClient, accessMode, csiSource, attachment, stagingPath)
	if err != nil {
		return "", err
	}

	return publishPath, nil
}

func (m *csiBlockMapper) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, podUID types.UID) error {
	return ioutil.MapBlockVolume(devicePath, globalMapPath, volumeMapPath, volumeMapName, podUID)
}

var _ volume.BlockVolumeUnmapper = &csiBlockMapper{}

// unpublishVolumeForBlock makes a block volume invisible via the publish path
func (m *csiBlockMapper) unpublishVolumeForBlock(ctx context.Context, csi csiClient, publishPath string) error {
	// Request to unmount bind mount to publishPath and to detach the device from the node.
	// If driver doesn't implement NodeUnstageVolume, detaching the device from the node is required, here.
	if err := csi.NodeUnpublishVolume(ctx, m.volumeID, publishPath); err != nil {
		klog.Error(log("blockMapper.unpublishVolumeForBlock failed: %v", err))
		return err
	}
	klog.V(4).Infof(log("blockMapper.unpublishVolumeForBlock NodeUnpublished successfully [%s]", publishPath))

	// Remove publishPath
	if err := os.Remove(publishPath); err != nil {
		klog.Error(log("blockMapper.unpublishVolumeForBlock failed to remove staging path after NodeUnpublishVolume() error [%s]: %v", publishPath, err))
		return err
	}

	return nil
}

// unstageVolumeForBlock makes a block volume invisible via the staging path
func (m *csiBlockMapper) unstageVolumeForBlock(ctx context.Context, csi csiClient, stagingPath string) error {
	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := hasStageUnstageCapability(ctx, csi)
	if err != nil {
		klog.Error(log("blockMapper.unstageVolumeForBlock failed to check STAGE_UNSTAGE_VOLUME capability: %v", err))
		return err
	}
	if !stageUnstageSet {
		klog.Infof(log("blockMapper.unstageVolumeForBlock STAGE_UNSTAGE_VOLUME capability not set. Skipping unstageVolumeForBlock ..."))
		return nil
	}

	// Request to unmount bind mount to stagingPath and to detach the device from the node.
	if err := csi.NodeUnstageVolume(ctx, m.volumeID, stagingPath); err != nil {
		klog.Errorf(log("blockMapper.unstageVolumeForBlock failed: %v", err))
		return err
	}
	klog.V(4).Infof(log("blockMapper.unstageVolumeForBlock NodeUnstageVolume successfully [%s]", stagingPath))

	// Remove stagingPath
	if err := os.Remove(stagingPath); err != nil {
		klog.Error(log("blockMapper.unstageVolumeForBlock failed to remove staging path after NodeUnstageVolume() error [%s]: %v", stagingPath, err))
		return err
	}

	return nil
}

// TearDownDevice removes traces of the SetUpDevice.
func (m *csiBlockMapper) TearDownDevice(globalMapPath, devicePath string) error {
	if !m.plugin.blockEnabled {
		return errors.New("CSIBlockVolume feature not enabled")
	}

	klog.V(4).Infof(log("unmapper.TearDownDevice(globalMapPath=%s; devicePath=%s)", globalMapPath, devicePath))

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	// Call NodeUnpublishVolume
	publishPath := m.getPublishPath()
	if _, err := os.Stat(publishPath); err != nil {
		if os.IsNotExist(err) {
			klog.V(4).Infof(log("blockMapper.TearDownDevice publishPath(%s) has already been deleted, skip calling NodeUnpublishVolume", publishPath))
		} else {
			return err
		}
	} else {
		err := m.unpublishVolumeForBlock(ctx, m.csiClient, publishPath)
		if err != nil {
			return err
		}
	}

	// Call NodeUnstageVolume
	stagingPath := m.getStagingPath()
	if _, err := os.Stat(stagingPath); err != nil {
		if os.IsNotExist(err) {
			klog.V(4).Infof(log("blockMapper.TearDownDevice stagingPath(%s) has already been deleted, skip calling NodeUnstageVolume", stagingPath))
		} else {
			return err
		}
	} else {
		err := m.unstageVolumeForBlock(ctx, m.csiClient, stagingPath)
		if err != nil {
			return err
		}
	}

	return nil
}
