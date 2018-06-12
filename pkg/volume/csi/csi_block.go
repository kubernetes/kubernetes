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

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/volume"
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

// GetGlobalMapPath returns a path (on the node) where the devicePath will be symlinked to
// Example: plugins/kubernetes.io/csi/volumeDevices/{volumeID}
func (m *csiBlockMapper) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	dir := getVolumeDevicePluginDir(spec.Name(), m.plugin.host)
	glog.V(4).Infof(log("blockMapper.GetGlobalMapPath = %s", dir))
	return dir, nil
}

// GetPodDeviceMapPath returns pod's device map path and volume name
// path: pods/{podUid}/volumeDevices/kubernetes.io~csi/, {volumeID}
func (m *csiBlockMapper) GetPodDeviceMapPath() (string, string) {
	path, specName := m.plugin.host.GetPodVolumeDeviceDir(m.podUID, csiPluginName), m.specName
	glog.V(4).Infof(log("blockMapper.GetPodDeviceMapPath = %s", path))
	return path, specName
}

// SetUpDevice ensures the device is attached returns path where the device is located.
func (m *csiBlockMapper) SetUpDevice() (string, error) {
	if !m.plugin.blockEnabled {
		return "", errors.New("CSIBlockVolume feature not enabled")
	}

	glog.V(4).Infof(log("blockMapper.SetupDevice called"))

	if m.spec == nil {
		glog.Error(log("blockMapper.Map spec is nil"))
		return "", fmt.Errorf("spec is nil")
	}
	csiSource, err := getCSISourceFromSpec(m.spec)
	if err != nil {
		glog.Error(log("blockMapper.SetupDevice failed to get CSI persistent source: %v", err))
		return "", err
	}

	globalMapPath, err := m.GetGlobalMapPath(m.spec)
	if err != nil {
		glog.Error(log("blockMapper.SetupDevice failed to get global map path: %v", err))
		return "", err
	}

	csi := m.csiClient
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := hasStageUnstageCapability(ctx, csi)
	if err != nil {
		glog.Error(log("blockMapper.SetupDevice failed to check STAGE_UNSTAGE_VOLUME capability: %v", err))
		return "", err
	}
	if !stageUnstageSet {
		glog.Infof(log("blockMapper.SetupDevice STAGE_UNSTAGE_VOLUME capability not set. Skipping MountDevice..."))
		return "", nil
	}

	// Start MountDevice
	nodeName := string(m.plugin.host.GetNodeName())
	attachID := getAttachmentName(csiSource.VolumeHandle, csiSource.Driver, nodeName)

	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	attachment, err := m.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		glog.Error(log("blockMapper.SetupDevice failed to get volume attachment [id=%v]: %v", attachID, err))
		return "", err
	}

	if attachment == nil {
		glog.Error(log("blockMapper.SetupDevice unable to find VolumeAttachment [id=%s]", attachID))
		return "", errors.New("no existing VolumeAttachment found")
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

	// create globalMapPath before call to NodeStageVolume
	if err := os.MkdirAll(globalMapPath, 0750); err != nil {
		glog.Error(log("blockMapper.SetupDevice failed to create dir %s: %v", globalMapPath, err))
		return "", err
	}
	glog.V(4).Info(log("blockMapper.SetupDevice created global device map path successfully [%s]", globalMapPath))

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := v1.ReadWriteOnce
	if m.spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = m.spec.PersistentVolume.Spec.AccessModes[0]
	}

	err = csi.NodeStageVolume(ctx,
		csiSource.VolumeHandle,
		publishVolumeInfo,
		globalMapPath,
		fsTypeBlockName,
		accessMode,
		nodeStageSecrets,
		csiSource.VolumeAttributes)

	if err != nil {
		glog.Error(log("blockMapper.SetupDevice failed: %v", err))
		if err := os.RemoveAll(globalMapPath); err != nil {
			glog.Error(log("blockMapper.SetupDevice failed to remove dir after a NodeStageVolume() error [%s]: %v", globalMapPath, err))
		}
		return "", err
	}

	glog.V(4).Infof(log("blockMapper.SetupDevice successfully requested NodeStageVolume [%s]", globalMapPath))
	return globalMapPath, nil
}

func (m *csiBlockMapper) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, podUID types.UID) error {
	if !m.plugin.blockEnabled {
		return errors.New("CSIBlockVolume feature not enabled")
	}

	glog.V(4).Infof(log("blockMapper.MapDevice mapping block device %s", devicePath))

	if m.spec == nil {
		glog.Error(log("blockMapper.MapDevice spec is nil"))
		return fmt.Errorf("spec is nil")
	}

	csiSource, err := getCSISourceFromSpec(m.spec)
	if err != nil {
		glog.Error(log("blockMapper.Map failed to get CSI persistent source: %v", err))
		return err
	}

	dir := filepath.Join(volumeMapPath, volumeMapName)
	csi := m.csiClient

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	nodeName := string(m.plugin.host.GetNodeName())
	attachID := getAttachmentName(csiSource.VolumeHandle, csiSource.Driver, nodeName)

	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	attachment, err := m.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
	if err != nil {
		glog.Error(log("blockMapper.MapDevice failed to get volume attachment [id=%v]: %v", attachID, err))
		return err
	}

	if attachment == nil {
		glog.Error(log("blockMapper.MapDevice unable to find VolumeAttachment [id=%s]", attachID))
		return errors.New("no existing VolumeAttachment found")
	}
	publishVolumeInfo := attachment.Status.AttachmentMetadata

	nodePublishSecrets := map[string]string{}
	if csiSource.NodePublishSecretRef != nil {
		nodePublishSecrets, err = getCredentialsFromSecret(m.k8s, csiSource.NodePublishSecretRef)
		if err != nil {
			glog.Errorf("blockMapper.MapDevice failed to get NodePublishSecretRef %s/%s: %v",
				csiSource.NodePublishSecretRef.Namespace, csiSource.NodePublishSecretRef.Name, err)
			return err
		}
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Error(log("blockMapper.MapDevice failed to create dir %#v:  %v", dir, err))
		return err
	}
	glog.V(4).Info(log("blockMapper.MapDevice created NodePublish path [%s]", dir))

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := v1.ReadWriteOnce
	if m.spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = m.spec.PersistentVolume.Spec.AccessModes[0]
	}

	err = csi.NodePublishVolume(
		ctx,
		m.volumeID,
		m.readOnly,
		globalMapPath,
		dir,
		accessMode,
		publishVolumeInfo,
		csiSource.VolumeAttributes,
		nodePublishSecrets,
		fsTypeBlockName,
	)

	if err != nil {
		glog.Errorf(log("blockMapper.MapDevice failed: %v", err))
		if err := os.RemoveAll(dir); err != nil {
			glog.Error(log("blockMapper.MapDevice failed to remove mount dir after a NodePublish() error [%s]: %v", dir, err))
		}
		return err
	}

	return nil
}

var _ volume.BlockVolumeUnmapper = &csiBlockMapper{}

// TearDownDevice removes traces of the SetUpDevice.
func (m *csiBlockMapper) TearDownDevice(globalMapPath, devicePath string) error {
	if !m.plugin.blockEnabled {
		return errors.New("CSIBlockVolume feature not enabled")
	}

	glog.V(4).Infof(log("unmapper.TearDownDevice(globalMapPath=%s; devicePath=%s)", globalMapPath, devicePath))

	csi := m.csiClient
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	// unmap global device map path
	if err := csi.NodeUnstageVolume(ctx, m.volumeID, globalMapPath); err != nil {
		glog.Errorf(log("blockMapper.TearDownDevice failed: %v", err))
		return err
	}
	glog.V(4).Infof(log("blockMapper.TearDownDevice NodeUnstageVolume successfully [%s]", globalMapPath))

	// request to remove pod volume map path also
	podVolumePath, volumeName := m.GetPodDeviceMapPath()
	podVolumeMapPath := filepath.Join(podVolumePath, volumeName)
	if err := csi.NodeUnpublishVolume(ctx, m.volumeID, podVolumeMapPath); err != nil {
		glog.Error(log("blockMapper.TearDownDevice failed: %v", err))
		return err
	}

	glog.V(4).Infof(log("blockMapper.TearDownDevice NodeUnpublished successfully [%s]", podVolumeMapPath))

	return nil
}
