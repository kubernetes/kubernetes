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
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"strings"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	RBDVolumePluginName           = "kubernetes.io/rbd"
	RBDDriverName                 = "rbd.csi.ceph.com"
	defaultAdminSecretNamespace   = "default"
	defaultImgFeatureVal          = "layering"
	defaultAdminUser              = "admin"
	defaultPoolVal                = "rbd"
	defaultIntreeImagePfx         = "kubernetes-dynamic-pvc-"
	defaultMigKey                 = "migration"
	defaultMigStaticVal           = "true"
	CSIRBDVolHandleAnnKey         = "rbd.csi.ceph.com/volume-handle"
	imgFeatureKey                 = "imageFeatures"
	imgFmtKey                     = "imageFormat"
	imgNameKey                    = "imageName"
	clusterIDKey                  = "clusterID"
	journalPoolKey                = "journalPool"
	poolKey                       = "pool"
	monsKey                       = "monitors"
	adminIDKey                    = "adminId"
	staticVolKey                  = "staticVolume"
	monsPfx                       = "mons-"
	imgPfx                        = "image-"
	migVolPfx                     = "mig"
	provSecretNameKey             = "csi.storage.k8s.io/provisioner-secret-name"
	nodeStageSecretNameKey        = "csi.storage.k8s.io/node-stage-secret-name"
	cntrlExpandSecretNameKey      = "csi.storage.k8s.io/controller-expand-secret-name"
	provSecretNamespaceKey        = "csi.storage.k8s.io/provisioner-secret-namespace"
	nodeStageSecretNamespaceKey   = "csi.storage.k8s.io/node-stage-secret-namespace"
	cntrlExpandSecretNamespaceKey = "csi.storage.k8s.io/controller-expand-secret-namespace"
)

var _ InTreePlugin = &rbdCSITranslator{}

type rbdCSITranslator struct{}

func NewRBDCSITranslator() InTreePlugin {
	return &rbdCSITranslator{}
}

// TranslateInTreeStorageClassToCSI takes in-tree storage class used by in-tree plugin
// and translates them to a storage class consumable by CSI plugin
func (p rbdCSITranslator) TranslateInTreeStorageClassToCSI(sc *storagev1.StorageClass) (*storagev1.StorageClass, error) {
	if sc == nil {
		return nil, fmt.Errorf("sc is nil")
	}

	var params = map[string]string{}

	fillDefaultSCParams(params)
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case fsTypeKey:
			params[csiFsTypeKey] = v
		case "imagefeatures":
			params[imgFeatureKey] = v
		case poolKey:
			params[poolKey] = v
		case "imageformat":
			params[imgFmtKey] = v
		case "adminid":
			params[adminIDKey] = v
		case "adminsecretname":
			params[provSecretNameKey] = v
			params[nodeStageSecretNameKey] = v
			params[cntrlExpandSecretNameKey] = v
		case "adminsecretnamespace":
			params[provSecretNamespaceKey] = v
			params[nodeStageSecretNamespaceKey] = v
			params[cntrlExpandSecretNamespaceKey] = v
		case monsKey:
			arr := strings.Split(v, ",")
			if len(arr) < 1 {
				return nil, fmt.Errorf("missing Ceph monitors")
			}
			params[monsKey] = v
			params[clusterIDKey] = fmt.Sprintf("%x", md5.Sum([]byte(v)))
		}
	}

	if params[provSecretNameKey] == "" {
		return nil, fmt.Errorf("missing Ceph admin secret name")
	}
	if params[monsKey] == "" {
		return nil, fmt.Errorf("missing Ceph monitors")
	}
	sc.Provisioner = RBDDriverName
	sc.Parameters = params
	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes an inline volume and will translate
// the in-tree inline volume source to a CSIPersistentVolumeSource
func (p rbdCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.RBD == nil {
		return nil, fmt.Errorf("volume is nil or RBDVolume not defined on volume")
	}

	var am v1.PersistentVolumeAccessMode
	if volume.RBD.ReadOnly {
		am = v1.ReadOnlyMany
	} else {
		am = v1.ReadWriteOnce
	}
	secRef := &v1.SecretReference{}
	if volume.RBD.SecretRef != nil {
		secRef.Name = volume.RBD.SecretRef.Name
		secRef.Namespace = podNamespace
	}
	volumeAttr := make(map[string]string)
	volumeAttr[clusterIDKey] = fmt.Sprintf("%x", md5.Sum([]byte(strings.Join(volume.RBD.CephMonitors, ","))))
	volumeAttr[poolKey] = defaultPoolVal
	if volume.RBD.RBDPool != "" {
		volumeAttr[poolKey] = volume.RBD.RBDPool
	}
	volumeAttr[staticVolKey] = defaultMigStaticVal
	volumeAttr[imgFeatureKey] = defaultImgFeatureVal
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("%s-%s", RBDDriverName, volume.RBD.RBDImage),
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:                    RBDDriverName,
					VolumeHandle:              volume.RBD.RBDImage,
					FSType:                    volume.RBD.FSType,
					VolumeAttributes:          volumeAttr,
					NodeStageSecretRef:        secRef,
					ControllerExpandSecretRef: secRef,
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{am},
		},
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a RBD persistent volume and will translate
// the in-tree pv source to a CSI Source
func (p rbdCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.RBD == nil {
		return nil, fmt.Errorf("pv is nil or RBD Volume not defined on pv")
	}
	var volID string
	volumeAttributes := make(map[string]string)

	if pv.Annotations[CSIRBDVolHandleAnnKey] != "" {
		volID = pv.Annotations[CSIRBDVolHandleAnnKey]
		volumeAttributes[clusterIDKey] = pv.Annotations[clusterIDKey]
	} else {
		mons := strings.Join(pv.Spec.RBD.CephMonitors, ",")
		pool := pv.Spec.RBD.RBDPool
		image := pv.Spec.RBD.RBDImage
		volumeAttributes[staticVolKey] = defaultMigStaticVal
		volumeAttributes[clusterIDKey] = fmt.Sprintf("%x", md5.Sum([]byte(mons)))
		volID = composeMigVolID(mons, pool, image)
	}

	err := fillVolAttrsForRequest(pv, volumeAttributes)
	if err != nil {
		return nil, err
	}
	if volumeAttributes[imgFeatureKey] == "" {
		volumeAttributes[imgFeatureKey] = defaultImgFeatureVal
	}
	var am v1.PersistentVolumeAccessMode
	if pv.Spec.RBD.ReadOnly {
		am = v1.ReadOnlyMany
	} else {
		am = v1.ReadWriteOnce
	}
	pv.Spec.AccessModes = []v1.PersistentVolumeAccessMode{am}
	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:                    RBDDriverName,
		FSType:                    pv.Spec.RBD.FSType,
		VolumeHandle:              volID,
		VolumeAttributes:          volumeAttributes,
		NodeStageSecretRef:        pv.Spec.RBD.SecretRef,
		ControllerExpandSecretRef: pv.Spec.RBD.SecretRef,
	}
	pv.Spec.RBD = nil
	pv.Spec.CSI = csiSource
	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with a CSI PersistentVolume Source and will translate
// it to an in-tree Persistent Volume Source for the in-tree volume
func (p rbdCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}
	var rbdImageName string
	monSlice := []string{""}
	csiSource := pv.Spec.CSI

	rbdImageName = csiSource.VolumeAttributes[imgNameKey]
	rbdPool := csiSource.VolumeAttributes[poolKey]
	radosUser := csiSource.VolumeAttributes[adminIDKey]
	if radosUser == "" {
		radosUser = defaultAdminUser
	}

	RBDSource := &v1.RBDPersistentVolumeSource{
		CephMonitors: monSlice,
		RBDImage:     rbdImageName,
		FSType:       csiSource.FSType,
		RBDPool:      rbdPool,
		RadosUser:    radosUser,
		ReadOnly:     csiSource.ReadOnly,
	}

	if pv.Annotations == nil {
		pv.Annotations = make(map[string]string)
	}
	fillAnnotationsFromCSISource(pv, csiSource)
	nodeSecret := csiSource.NodeStageSecretRef
	if nodeSecret != nil {
		RBDSource.SecretRef = &v1.SecretReference{Name: nodeSecret.Name, Namespace: nodeSecret.Namespace}
	}
	pv.Spec.CSI = nil
	pv.Spec.RBD = RBDSource

	return pv, nil
}

// CanSupport tests whether the plugin supports a given persistent volume
// specification from the API.
func (p rbdCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.RBD != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.
func (p rbdCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.RBD != nil
}

// GetInTreePluginName returns the in-tree plugin name this migrates
func (p rbdCSITranslator) GetInTreePluginName() string {
	return RBDVolumePluginName
}

// GetCSIPluginName returns the name of the CSI plugin that supersedes the in-tree plugin
func (p rbdCSITranslator) GetCSIPluginName() string {
	return RBDDriverName
}

// RepairVolumeHandle generates a correct volume handle based on node ID information.
func (p rbdCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
}

// fillDefaultSCParams fills some sc parameters with default values
func fillDefaultSCParams(params map[string]string) {
	params[defaultMigKey] = defaultMigStaticVal
	params[poolKey] = defaultPoolVal
	params[provSecretNamespaceKey] = defaultAdminSecretNamespace
	params[cntrlExpandSecretNamespaceKey] = defaultAdminSecretNamespace
	params[nodeStageSecretNamespaceKey] = defaultAdminSecretNamespace
}

// composeMigVolID composes migration handle for RBD PV
// mig_mons-afcca55bc1bdd3f479be1e8281c13ab1_image-e0b45b52-7e09-47d3-8f1b-806995fa4412_7265706c696361706f6f6c
func composeMigVolID(mons string, pool string, image string) string {
	clusterIDInHandle := md5.Sum([]byte(mons))
	clusterField := monsPfx + fmt.Sprintf("%x", clusterIDInHandle)
	poolHashInHandle := hex.EncodeToString([]byte(pool))
	imageHashInHandle := strings.Split(image, defaultIntreeImagePfx)[1]
	imageField := imgPfx + imageHashInHandle
	volHash := strings.Join([]string{migVolPfx, clusterField, imageField, poolHashInHandle}, "_")
	return volHash
}

// fillVolAttrsForRequest fill the volume attributes for node operations
func fillVolAttrsForRequest(pv *v1.PersistentVolume, volumeAttributes map[string]string) error {
	if pv == nil || pv.Spec.RBD == nil {
		return fmt.Errorf("pv is nil or RBD Volume not defined on pv")
	}
	volumeAttributes[imgNameKey] = pv.Spec.RBD.RBDImage
	volumeAttributes[poolKey] = pv.Spec.RBD.RBDPool
	volumeAttributes[imgFeatureKey] = pv.Annotations[imgFeatureKey]
	volumeAttributes[imgFmtKey] = pv.Annotations[imgFmtKey]
	volumeAttributes[journalPoolKey] = pv.Annotations[journalPoolKey]
	volumeAttributes[defaultMigKey] = defaultMigStaticVal
	volumeAttributes["tryOtherMounters"] = defaultMigStaticVal
	return nil
}

// fillAnnotationsFromCSISource capture required information from csi source
func fillAnnotationsFromCSISource(pv *v1.PersistentVolume, csiSource *v1.CSIPersistentVolumeSource) {
	pv.Annotations[CSIRBDVolHandleAnnKey] = csiSource.VolumeHandle
	pv.Annotations[clusterIDKey] = csiSource.VolumeAttributes[clusterIDKey]
	pv.Annotations[journalPoolKey] = csiSource.VolumeAttributes[journalPoolKey]
	pv.Annotations[imgFeatureKey] = csiSource.VolumeAttributes[imgFeatureKey]
	pv.Annotations[imgFmtKey] = csiSource.VolumeAttributes[imgFmtKey]
}
