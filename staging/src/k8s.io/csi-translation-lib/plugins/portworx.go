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
	"strings"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

const (
	PortworxVolumePluginName = "kubernetes.io/portworx-volume"
	PortworxDriverName       = "pxd.portworx.com"

	OpenStorageAuthSecretNameKey      = "openstorage.io/auth-secret-name"
	OpenStorageAuthSecretNamespaceKey = "openstorage.io/auth-secret-namespace"

	csiParameterPrefix = "csi.storage.k8s.io/"

	prefixedProvisionerSecretNameKey      = csiParameterPrefix + "provisioner-secret-name"
	prefixedProvisionerSecretNamespaceKey = csiParameterPrefix + "provisioner-secret-namespace"

	prefixedControllerPublishSecretNameKey      = csiParameterPrefix + "controller-publish-secret-name"
	prefixedControllerPublishSecretNamespaceKey = csiParameterPrefix + "controller-publish-secret-namespace"

	prefixedNodeStageSecretNameKey      = csiParameterPrefix + "node-stage-secret-name"
	prefixedNodeStageSecretNamespaceKey = csiParameterPrefix + "node-stage-secret-namespace"

	prefixedNodePublishSecretNameKey      = csiParameterPrefix + "node-publish-secret-name"
	prefixedNodePublishSecretNamespaceKey = csiParameterPrefix + "node-publish-secret-namespace"

	prefixedControllerExpandSecretNameKey      = csiParameterPrefix + "controller-expand-secret-name"
	prefixedControllerExpandSecretNamespaceKey = csiParameterPrefix + "controller-expand-secret-namespace"

	prefixedNodeExpandSecretNameKey      = csiParameterPrefix + "node-expand-secret-name"
	prefixedNodeExpandSecretNamespaceKey = csiParameterPrefix + "node-expand-secret-namespace"
)

var _ InTreePlugin = &portworxCSITranslator{}

type portworxCSITranslator struct{}

func NewPortworxCSITranslator() InTreePlugin {
	return &portworxCSITranslator{}
}

// TranslateInTreeStorageClassToCSI takes in-tree storage class used by in-tree plugin
// and translates them to a storageclass consumable by CSI plugin
func (p portworxCSITranslator) TranslateInTreeStorageClassToCSI(logger klog.Logger, sc *storagev1.StorageClass) (*storagev1.StorageClass, error) {
	if sc == nil {
		return nil, fmt.Errorf("sc is nil")
	}

	var params = map[string]string{}
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case OpenStorageAuthSecretNameKey:
			params[prefixedProvisionerSecretNameKey] = v
			params[prefixedControllerPublishSecretNameKey] = v
			params[prefixedNodePublishSecretNameKey] = v
			params[prefixedNodeStageSecretNameKey] = v
			params[prefixedControllerExpandSecretNameKey] = v
			params[prefixedNodeExpandSecretNameKey] = v
		case OpenStorageAuthSecretNamespaceKey:
			params[prefixedProvisionerSecretNamespaceKey] = v
			params[prefixedControllerPublishSecretNamespaceKey] = v
			params[prefixedNodePublishSecretNamespaceKey] = v
			params[prefixedNodeStageSecretNamespaceKey] = v
			params[prefixedControllerExpandSecretNamespaceKey] = v
			params[prefixedNodeExpandSecretNamespaceKey] = v
		default:
			// All other parameters can be copied as is
			params[k] = v
		}
	}
	if len(params) > 0 {
		sc.Parameters = params
	}
	sc.Provisioner = PortworxDriverName

	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a inline volume and will translate
// the in-tree inline volume source to a CSIPersistentVolumeSource
func (p portworxCSITranslator) TranslateInTreeInlineVolumeToCSI(logger klog.Logger, volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
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
func (p portworxCSITranslator) TranslateInTreePVToCSI(logger klog.Logger, pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.PortworxVolume == nil {
		return nil, fmt.Errorf("pv is nil or PortworxVolume not defined on pv")
	}
	var secretRef *v1.SecretReference

	if metav1.HasAnnotation(pv.ObjectMeta, OpenStorageAuthSecretNameKey) &&
		metav1.HasAnnotation(pv.ObjectMeta, OpenStorageAuthSecretNamespaceKey) {
		secretRef = &v1.SecretReference{
			Name:      pv.Annotations[OpenStorageAuthSecretNameKey],
			Namespace: pv.Annotations[OpenStorageAuthSecretNamespaceKey],
		}
	}

	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:                     PortworxDriverName,
		VolumeHandle:               pv.Spec.PortworxVolume.VolumeID,
		FSType:                     pv.Spec.PortworxVolume.FSType,
		VolumeAttributes:           make(map[string]string), // copy access mode
		ControllerPublishSecretRef: secretRef,
		NodeStageSecretRef:         secretRef,
		NodePublishSecretRef:       secretRef,
		ControllerExpandSecretRef:  secretRef,
		NodeExpandSecretRef:        secretRef,
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
