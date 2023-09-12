/*
Copyright 2021 The Kubernetes Authors.

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

package plugins

import (
	"fmt"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	PortworxVolumePluginName = "kubernetes.io/portworx-volume"
	PortworxDriverName       = "pxd.portworx.com"
)

var _ InTreePlugin = &portworxCSITranslator{}

type portworxCSITranslator struct{}

func NewPortworxCSITranslator() InTreePlugin {
	return &portworxCSITranslator{}
}

// TranslateInTreeStorageClassToCSI takes in-tree storage class used by in-tree plugin
// and translates them to a storageclass consumable by CSI plugin
func (p portworxCSITranslator) TranslateInTreeStorageClassToCSI(sc *storagev1.StorageClass) (*storagev1.StorageClass, error) {
	if sc == nil {
		return nil, fmt.Errorf("sc is nil")
	}
	sc.Provisioner = PortworxDriverName
	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a inline volume and will translate
// the in-tree inline volume source to a CSIPersistentVolumeSource
func (p portworxCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.PortworxVolume == nil {
		return nil, fmt.Errorf("volume is nil or PortworxVolume not defined on volume")
	}

	var am v1.PersistentVolumeAccessMode
	if volume.PortworxVolume.ReadOnly {
		am = v1.ReadOnlyMany
	} else {
		am = v1.ReadWriteOnce
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("%s-%s", PortworxDriverName, volume.PortworxVolume.VolumeID),
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:           PortworxDriverName,
					VolumeHandle:     volume.PortworxVolume.VolumeID,
					FSType:           volume.PortworxVolume.FSType,
					VolumeAttributes: make(map[string]string),
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{am},
		},
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a Portworx persistent volume and will translate
// the in-tree pv source to a CSI Source
func (p portworxCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.PortworxVolume == nil {
		return nil, fmt.Errorf("pv is nil or PortworxVolume not defined on pv")
	}
	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:           PortworxDriverName,
		VolumeHandle:     pv.Spec.PortworxVolume.VolumeID,
		FSType:           pv.Spec.PortworxVolume.FSType,
		VolumeAttributes: make(map[string]string), // copy access mode
	}
	pv.Spec.PortworxVolume = nil
	pv.Spec.CSI = csiSource

	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with a CSI PersistentVolume Source and will translate
// it to a in-tree Persistent Volume Source for the in-tree volume
func (p portworxCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}
	csiSource := pv.Spec.CSI

	portworxSource := &v1.PortworxVolumeSource{
		VolumeID: csiSource.VolumeHandle,
		FSType:   csiSource.FSType,
		ReadOnly: csiSource.ReadOnly,
	}
	pv.Spec.CSI = nil
	pv.Spec.PortworxVolume = portworxSource

	return pv, nil
}

// CanSupport tests whether the plugin supports a given persistent volume
// specification from the API.
func (p portworxCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.PortworxVolume != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.
func (p portworxCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.PortworxVolume != nil
}

// GetInTreePluginName returns the in-tree plugin name this migrates
func (p portworxCSITranslator) GetInTreePluginName() string {
	return PortworxVolumePluginName
}

// GetCSIPluginName returns the name of the CSI plugin that supersedes the in-tree plugin
func (p portworxCSITranslator) GetCSIPluginName() string {
	return PortworxDriverName
}

// RepairVolumeHandle generates a correct volume handle based on node ID information.
func (p portworxCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
}
