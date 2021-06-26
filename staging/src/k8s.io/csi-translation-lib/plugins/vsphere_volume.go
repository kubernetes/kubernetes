/*
Copyright 2020 The Kubernetes Authors.

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
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

const (
	// VSphereDriverName is the name of the CSI driver for vSphere Volume
	VSphereDriverName = "csi.vsphere.vmware.com"
	// VSphereInTreePluginName is the name of the in-tree plugin for vSphere Volume
	VSphereInTreePluginName = "kubernetes.io/vsphere-volume"

	// paramStoragePolicyName used to supply SPBM Policy name for Volume provisioning
	paramStoragePolicyName = "storagepolicyname"

	// This param is used to tell Driver to return volumePath and not VolumeID
	// in-tree vSphere plugin does not understand volume id, it uses volumePath
	paramcsiMigration = "csimigration"

	// This param is used to supply datastore name for Volume provisioning
	paramDatastore = "datastore-migrationparam"

	// This param supplies disk foramt (thin, thick, zeoredthick) for Volume provisioning
	paramDiskFormat = "diskformat-migrationparam"

	// vSAN Policy Parameters
	paramHostFailuresToTolerate = "hostfailurestotolerate-migrationparam"
	paramForceProvisioning      = "forceprovisioning-migrationparam"
	paramCacheReservation       = "cachereservation-migrationparam"
	paramDiskstripes            = "diskstripes-migrationparam"
	paramObjectspacereservation = "objectspacereservation-migrationparam"
	paramIopslimit              = "iopslimit-migrationparam"

	// AttributeInitialVolumeFilepath represents the path of volume where volume is created
	AttributeInitialVolumeFilepath = "initialvolumefilepath"
)

var _ InTreePlugin = &vSphereCSITranslator{}

// vSphereCSITranslator handles translation of PV spec from In-tree vSphere Volume to vSphere CSI
type vSphereCSITranslator struct{}

// NewvSphereCSITranslator returns a new instance of vSphereCSITranslator
func NewvSphereCSITranslator() InTreePlugin {
	return &vSphereCSITranslator{}
}

// TranslateInTreeStorageClassToCSI translates InTree vSphere storage class parameters to CSI storage class
func (t *vSphereCSITranslator) TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error) {
	if sc == nil {
		return nil, fmt.Errorf("sc is nil")
	}
	var params = map[string]string{}
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case fsTypeKey:
			params[csiFsTypeKey] = v
		case paramStoragePolicyName:
			params[paramStoragePolicyName] = v
		case "datastore":
			params[paramDatastore] = v
		case "diskformat":
			params[paramDiskFormat] = v
		case "hostfailurestotolerate":
			params[paramHostFailuresToTolerate] = v
		case "forceprovisioning":
			params[paramForceProvisioning] = v
		case "cachereservation":
			params[paramCacheReservation] = v
		case "diskstripes":
			params[paramDiskstripes] = v
		case "objectspacereservation":
			params[paramObjectspacereservation] = v
		case "iopslimit":
			params[paramIopslimit] = v
		default:
			klog.V(2).Infof("StorageClass parameter [name:%q, value:%q] is not supported", k, v)
		}
	}

	// This helps vSphere CSI driver to identify in-tree provisioner request vs CSI provisioner request
	// When this is true, Driver returns initialvolumefilepath in the VolumeContext, which is
	// used in TranslateCSIPVToInTree
	params[paramcsiMigration] = "true"
	// Note: sc.AllowedTopologies for Topology based volume provisioning will be supplied as it is.
	sc.Parameters = params
	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with VsphereVolume set from in-tree
// and converts the VsphereVolume source to a CSIPersistentVolumeSource
func (t *vSphereCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.VsphereVolume == nil {
		return nil, fmt.Errorf("volume is nil or VsphereVolume not defined on volume")
	}
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			// Must be unique per disk as it is used as the unique part of the
			// staging path
			Name: fmt.Sprintf("%s-%s", VSphereDriverName, volume.VsphereVolume.VolumePath),
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:           VSphereDriverName,
					VolumeHandle:     volume.VsphereVolume.VolumePath,
					FSType:           volume.VsphereVolume.FSType,
					VolumeAttributes: make(map[string]string),
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
	if volume.VsphereVolume.StoragePolicyName != "" {
		pv.Spec.CSI.VolumeAttributes[paramStoragePolicyName] = pv.Spec.VsphereVolume.StoragePolicyName
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with VsphereVolume set from in-tree
// and converts the VsphereVolume source to a CSIPersistentVolumeSource
func (t *vSphereCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.VsphereVolume == nil {
		return nil, fmt.Errorf("pv is nil or VsphereVolume not defined on pv")
	}
	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:           VSphereDriverName,
		VolumeHandle:     pv.Spec.VsphereVolume.VolumePath,
		FSType:           pv.Spec.VsphereVolume.FSType,
		VolumeAttributes: make(map[string]string),
	}
	if pv.Spec.VsphereVolume.StoragePolicyName != "" {
		csiSource.VolumeAttributes[paramStoragePolicyName] = pv.Spec.VsphereVolume.StoragePolicyName
	}
	pv.Spec.VsphereVolume = nil
	pv.Spec.CSI = csiSource
	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with CSIPersistentVolumeSource set and
// translates the vSphere CSI source to a vSphereVolume source.
func (t *vSphereCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}
	csiSource := pv.Spec.CSI
	vsphereVirtualDiskVolumeSource := &v1.VsphereVirtualDiskVolumeSource{
		FSType: csiSource.FSType,
	}
	volumeFilePath, ok := csiSource.VolumeAttributes[AttributeInitialVolumeFilepath]
	if ok {
		vsphereVirtualDiskVolumeSource.VolumePath = volumeFilePath
	}
	pv.Spec.CSI = nil
	pv.Spec.VsphereVolume = vsphereVirtualDiskVolumeSource
	return pv, nil
}

// CanSupport tests whether the plugin supports a given persistent volume
// specification from the API.
func (t *vSphereCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.VsphereVolume != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.
func (t *vSphereCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.VsphereVolume != nil
}

// GetInTreePluginName returns the name of the in-tree plugin driver
func (t *vSphereCSITranslator) GetInTreePluginName() string {
	return VSphereInTreePluginName
}

// GetCSIPluginName returns the name of the CSI plugin
func (t *vSphereCSITranslator) GetCSIPluginName() string {
	return VSphereDriverName
}

// RepairVolumeHandle is needed in VerifyVolumesAttached on the external attacher when we need to do strict volume
// handle matching to check VolumeAttachment attached status.
// vSphere volume does not need patch to help verify whether that volume is attached.
func (t *vSphereCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
}
