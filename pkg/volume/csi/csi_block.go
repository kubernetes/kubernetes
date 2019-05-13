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
	"path/filepath"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/volume"
	ioutil "k8s.io/kubernetes/pkg/volume/util"
	utilstrings "k8s.io/utils/strings"
)

type csiBlockMapper struct {
	csiClientGetter
	k8s        kubernetes.Interface
	plugin     *csiPlugin
	driverName csiDriverName
	specName   string
	volumeID   string
	readOnly   bool
	spec       *volume.Spec
	podUID     types.UID
	volumeInfo map[string]string
}

var _ volume.BlockVolumeMapper = &csiBlockMapper{}

// GetGlobalMapPath returns a global map path (on the node) to a device file which will be symlinked to
// Example: plugins/kubernetes.io/csi/volumeDevices/{pvname}/dev
func (m *csiBlockMapper) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	dir := getVolumeDevicePluginDir(spec.Name(), m.plugin.host)
	klog.V(4).Infof(log("blockMapper.GetGlobalMapPath = %s", dir))
	return dir, nil
}

// getStagingPath returns a staging path for a directory (on the node) that should be used on NodeStageVolume/NodeUnstageVolume
// Example: plugins/kubernetes.io/csi/volumeDevices/staging/{pvname}
func (m *csiBlockMapper) getStagingPath() string {
	sanitizedSpecVolID := utilstrings.EscapeQualifiedName(m.specName)
	return filepath.Join(m.plugin.host.GetVolumeDevicePluginDir(CSIPluginName), "staging", sanitizedSpecVolID)
}

// getPublishPath returns a publish path for a file (on the node) that should be used on NodePublishVolume/NodeUnpublishVolume
// Example: plugins/kubernetes.io/csi/volumeDevices/publish/{pvname}
func (m *csiBlockMapper) getPublishPath() string {
	sanitizedSpecVolID := utilstrings.EscapeQualifiedName(m.specName)
	return filepath.Join(m.plugin.host.GetVolumeDevicePluginDir(CSIPluginName), "publish", sanitizedSpecVolID)
}

// GetPodDeviceMapPath returns pod's device file which will be mapped to a volume
// returns: pods/{podUid}/volumeDevices/kubernetes.io~csi, {pvname}
func (m *csiBlockMapper) GetPodDeviceMapPath() (string, string) {
	path := m.plugin.host.GetPodVolumeDeviceDir(m.podUID, utilstrings.EscapeQualifiedName(CSIPluginName))
	specName := m.specName
	klog.V(4).Infof(log("blockMapper.GetPodDeviceMapPath [path=%s; name=%s]", path, specName))
	return path, specName
}

// stageVolumeForBlock stages a block volume to stagingPath
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
	stageUnstageSet, err := csi.NodeSupportsStageUnstage(ctx)
	if err != nil {
		klog.Error(log("blockMapper.stageVolumeForBlock failed to check STAGE_UNSTAGE_VOLUME capability: %v", err))
		return "", err
	}
	if !stageUnstageSet {
		klog.Infof(log("blockMapper.stageVolumeForBlock STAGE_UNSTAGE_VOLUME capability not set. Skipping MountDevice..."))
		return "", nil
	}

	publishVolumeInfo := attachment.Status.AttachmentMetadata

	nodeStageSecrets := map[string]string{}
	if csiSource.NodeStageSecretRef != nil {
		nodeStageSecrets, err = getCredentialsFromSecret(m.k8s, csiSource.NodeStageSecretRef)
		if err != nil {
			return "", fmt.Errorf("failed to get NodeStageSecretRef %s/%s: %v",
				csiSource.NodeStageSecretRef.Namespace, csiSource.NodeStageSecretRef.Name, err)
		}
	}

	// Creating a stagingPath directory before call to NodeStageVolume
	if err := os.MkdirAll(stagingPath, 0750); err != nil {
		klog.Error(log("blockMapper.stageVolumeForBlock failed to create dir %s: %v", stagingPath, err))
		return "", err
	}
	klog.V(4).Info(log("blockMapper.stageVolumeForBlock created stagingPath directory successfully [%s]", stagingPath))

	// Request to stage a block volume to stagingPath.
	// Expected implementation for driver is creating driver specific resource on stagingPath and
	// attaching the block volume to the node.
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

// publishVolumeForBlock publishes a block volume to publishPath
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
	// Setup a parent directory for publishPath before call to NodePublishVolume
	publishDir := filepath.Dir(publishPath)
	if err := os.MkdirAll(publishDir, 0750); err != nil {
		klog.Error(log("blockMapper.publishVolumeForBlock failed to create dir %s:  %v", publishDir, err))
		return "", err
	}
	klog.V(4).Info(log("blockMapper.publishVolumeForBlock created directory for publishPath successfully [%s]", publishDir))

	// Request to publish a block volume to publishPath.
	// Expectation for driver is to place a block volume on the publishPath, by bind-mounting the device file on the publishPath or
	// creating device file on the publishPath.
	// Parent directory for publishPath is created by k8s, but driver is responsible for creating publishPath itself.
	// If driver doesn't implement NodeStageVolume, attaching the block volume to the node may be done, here.
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
	attachment, err := m.k8s.StorageV1().VolumeAttachments().Get(attachID, meta.GetOptions{})
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

	csiClient, err := m.csiClientGetter.Get()
	if err != nil {
		klog.Error(log("blockMapper.SetUpDevice failed to get CSI client: %v", err))
		return "", err
	}

	// Call NodeStageVolume
	stagingPath, err := m.stageVolumeForBlock(ctx, csiClient, accessMode, csiSource, attachment)
	if err != nil {
		return "", err
	}

	// Call NodePublishVolume
	publishPath, err := m.publishVolumeForBlock(ctx, csiClient, accessMode, csiSource, attachment, stagingPath)
	if err != nil {
		return "", err
	}

	return publishPath, nil
}

func (m *csiBlockMapper) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, podUID types.UID) error {
	return ioutil.MapBlockVolume(devicePath, globalMapPath, volumeMapPath, volumeMapName, podUID)
}

var _ volume.BlockVolumeUnmapper = &csiBlockMapper{}

// unpublishVolumeForBlock unpublishes a block volume from publishPath
func (m *csiBlockMapper) unpublishVolumeForBlock(ctx context.Context, csi csiClient, publishPath string) error {
	// Request to unpublish a block volume from publishPath.
	// Expectation for driver is to remove block volume from the publishPath, by unmounting bind-mounted device file
	// or deleting device file.
	// Driver is responsible for deleting publishPath itself.
	// If driver doesn't implement NodeUnstageVolume, detaching the block volume from the node may be done, here.
	if err := csi.NodeUnpublishVolume(ctx, m.volumeID, publishPath); err != nil {
		klog.Error(log("blockMapper.unpublishVolumeForBlock failed: %v", err))
		return err
	}
	klog.V(4).Infof(log("blockMapper.unpublishVolumeForBlock NodeUnpublished successfully [%s]", publishPath))

	return nil
}

// unstageVolumeForBlock unstages a block volume from stagingPath
func (m *csiBlockMapper) unstageVolumeForBlock(ctx context.Context, csi csiClient, stagingPath string) error {
	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := csi.NodeSupportsStageUnstage(ctx)
	if err != nil {
		klog.Error(log("blockMapper.unstageVolumeForBlock failed to check STAGE_UNSTAGE_VOLUME capability: %v", err))
		return err
	}
	if !stageUnstageSet {
		klog.Infof(log("blockMapper.unstageVolumeForBlock STAGE_UNSTAGE_VOLUME capability not set. Skipping unstageVolumeForBlock ..."))
		return nil
	}

	// Request to unstage a block volume from stagingPath.
	// Expected implementation for driver is removing driver specific resource in stagingPath and
	// detaching the block volume from the node.
	if err := csi.NodeUnstageVolume(ctx, m.volumeID, stagingPath); err != nil {
		klog.Errorf(log("blockMapper.unstageVolumeForBlock failed: %v", err))
		return err
	}
	klog.V(4).Infof(log("blockMapper.unstageVolumeForBlock NodeUnstageVolume successfully [%s]", stagingPath))

	// Remove stagingPath directory and its contents
	if err := os.RemoveAll(stagingPath); err != nil {
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

	csiClient, err := m.csiClientGetter.Get()
	if err != nil {
		klog.Error(log("blockMapper.TearDownDevice failed to get CSI client: %v", err))
		return err
	}

	// Call NodeUnpublishVolume
	publishPath := m.getPublishPath()
	if _, err := os.Stat(publishPath); err != nil {
		if os.IsNotExist(err) {
			klog.V(4).Infof(log("blockMapper.TearDownDevice publishPath(%s) has already been deleted, skip calling NodeUnpublishVolume", publishPath))
		} else {
			return err
		}
	} else {
		err := m.unpublishVolumeForBlock(ctx, csiClient, publishPath)
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
		err := m.unstageVolumeForBlock(ctx, csiClient, stagingPath)
		if err != nil {
			return err
		}
	}

	return nil
}
