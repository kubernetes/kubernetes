/*
Copyright 2022 The Kubernetes Authors.
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
	"fmt"
	"strings"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	CephFSVolumePluginName            = "kubernetes.io/cephfs"
	CephFSDriverName                  = "cephfs.csi.ceph.com"
	defaultCephFSAdminSecretNamespace = "default"
	defaultCephFSAdminUser            = "admin"
	defaultFsNameKey                  = "fsName"
	defaultFSNameVal                  = "myfs"
	defaultRootPathKey                = "rootPath"
	defaultRootPathVal                = "/"
	defaultCephFSMigKey               = "migration"
	defaultCephFSMigStaticVal         = "true"
	CSICephFSVolHandleAnnKey          = "cephfs.csi.ceph.com/volume-handle"
	cephfsclusterIDKey                = "clusterID"
	cephfsmonsKey                     = "monitors"
	cephfsadminIDKey                  = "adminId"
	cephfsstaticVolKey                = "staticVolume"
	cephfsmonsPfx                     = "mons-"
	cephfspathPfx                     = "path-"
	cephfsnodeStageSecretNameKey      = "csi.storage.k8s.io/node-stage-secret-name"
	cephfsnodeStageSecretNamespaceKey = "csi.storage.k8s.io/node-stage-secret-namespace"
)

var _ InTreePlugin = &cephfsCSITranslator{}

type cephfsCSITranslator struct{}

func NewCephFSCSITranslator() InTreePlugin {
	return &cephfsCSITranslator{}
}

// TranslateInTreeStorageClassToCSI takes in-tree storage class used by in-tree plugin
// and translates them to a storage class consumable by CSI plugin, but in this case its not required.
func (p cephfsCSITranslator) TranslateInTreeStorageClassToCSI(sc *storagev1.StorageClass) (*storagev1.StorageClass, error) {
	if sc == nil {
		return nil, fmt.Errorf("sc is nil")
	}

	var params = map[string]string{}

	fillDefaultSCParamsForCephFS(params)
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case "adminid":
			params[adminIDKey] = v
		case "adminsecretname":
			params[nodeStageSecretNameKey] = v
		case "adminsecretnamespace":
			params[nodeStageSecretNamespaceKey] = v
		case monsKey:
			arr := strings.Split(v, ",")
			if len(arr) < 1 {
				return nil, fmt.Errorf("missing Ceph monitors")
			}
			params[monsKey] = v
			params[clusterIDKey] = fmt.Sprintf("%x", md5.Sum([]byte(v)))
		}
	}

	if params[monsKey] == "" {
		return nil, fmt.Errorf("missing Ceph monitors")
	}
	sc.Provisioner = CephFSDriverName
	sc.Parameters = params
	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes an inline volume and will translate
// the in-tree inline volume source to a CSIPersistentVolumeSource
func (p cephfsCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.CephFS == nil {
		return nil, fmt.Errorf("volume is nil or CephFSVolume not defined on volume")
	}

	var am v1.PersistentVolumeAccessMode
	if volume.CephFS.ReadOnly {
		am = v1.ReadOnlyMany
	} else {
		am = v1.ReadWriteMany
	}
	secRef := &v1.SecretReference{}
	if volume.CephFS.SecretRef != nil {
		secRef.Name = volume.CephFS.SecretRef.Name
		secRef.Namespace = podNamespace
	}
	volumeAttr := make(map[string]string)
	volumeAttr[clusterIDKey] = fmt.Sprintf("%x", md5.Sum([]byte(strings.Join(volume.CephFS.Monitors, ","))))
	volumeAttr[defaultFsNameKey] = defaultFSNameVal
	volumeAttr[staticVolKey] = defaultMigStaticVal
	volumeAttr[defaultRootPathKey] = defaultRootPathVal
	if volume.CephFS.Path != "" {
		//todo: append the prefix after checking
		volumeAttr[defaultRootPathKey] = volume.CephFS.Path
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("%s-%s", CephFSDriverName, volume.CephFS.Path),
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:             CephFSDriverName,
					VolumeHandle:       volume.CephFS.Path,
					VolumeAttributes:   volumeAttr,
					NodeStageSecretRef: secRef,
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{am},
		},
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a CephFS persistent volume and will translate
// the in-tree pv source to a CSI Source
func (p cephfsCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CephFS == nil {
		return nil, fmt.Errorf("pv is nil or CephFS Volume not defined on pv")
	}
	var volID string
	volumeAttributes := make(map[string]string)

	if pv.Annotations[CSICephFSVolHandleAnnKey] != "" {
		volID = pv.Annotations[CSICephFSVolHandleAnnKey]
		volumeAttributes[clusterIDKey] = pv.Annotations[clusterIDKey]
	} else {
		mons := strings.Join(pv.Spec.CephFS.Monitors, ",")
		path := pv.Spec.CephFS.Path
		volumeAttributes[staticVolKey] = defaultMigStaticVal
		volumeAttributes[clusterIDKey] = fmt.Sprintf("%x", md5.Sum([]byte(mons)))
		volID = composeMigVolIDForCephFs(mons, path)
	}

	err := fillVolAttrsForCephFSRequest(pv, volumeAttributes)
	if err != nil {
		return nil, err
	}

	var am v1.PersistentVolumeAccessMode
	if pv.Spec.CephFS.ReadOnly {
		am = v1.ReadOnlyMany
	} else {
		am = v1.ReadWriteOnce
	}
	pv.Spec.AccessModes = []v1.PersistentVolumeAccessMode{am}
	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:                    CephFSDriverName,
		VolumeHandle:              volID,
		VolumeAttributes:          volumeAttributes,
		NodeStageSecretRef:        pv.Spec.CephFS.SecretRef,
		ControllerExpandSecretRef: pv.Spec.CephFS.SecretRef,
	}
	pv.Spec.CephFS = nil
	pv.Spec.CSI = csiSource
	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with a CSI PersistentVolume Source and will translate
// it to an in-tree Persistent Volume Source for the in-tree volume
func (p cephfsCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}
	var cephfsVolumeName string
	var monSlice []string
	csiSource := pv.Spec.CSI

	cephfsVolumeName = csiSource.VolumeAttributes[defaultRootPathKey]
	radosUser := csiSource.VolumeAttributes[adminIDKey]
	if radosUser == "" {
		radosUser = defaultAdminUser
	}

	CephFSSource := &v1.CephFSPersistentVolumeSource{
		Monitors: monSlice,
		Path:     cephfsVolumeName,
		User:     radosUser,
		ReadOnly: csiSource.ReadOnly,
	}

	if pv.Annotations == nil {
		pv.Annotations = make(map[string]string)
	}
	fillAnnotationsFromCephFSCSISource(pv, csiSource)
	nodeSecret := csiSource.NodeStageSecretRef
	if nodeSecret != nil {
		CephFSSource.SecretRef = &v1.SecretReference{Name: nodeSecret.Name, Namespace: nodeSecret.Namespace}
	}
	pv.Spec.CSI = nil
	pv.Spec.CephFS = CephFSSource

	return pv, nil
}

// CanSupport tests whether the plugin supports a given persistent volume
// specification from the API.
func (p cephfsCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.CephFS != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.
func (p cephfsCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.CephFS != nil
}

// GetInTreePluginName returns the in-tree plugin name this migrates
func (p cephfsCSITranslator) GetInTreePluginName() string {
	return CephFSVolumePluginName
}

// GetCSIPluginName returns the name of the CSI plugin that supersedes the in-tree plugin
func (p cephfsCSITranslator) GetCSIPluginName() string {
	return CephFSDriverName
}

// RepairVolumeHandle generates a correct volume handle based on node ID information.
func (p cephfsCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
}

// fillDefaultSCParams fills some sc parameters with default values
func fillDefaultSCParamsForCephFS(params map[string]string) {
	params[defaultMigKey] = defaultMigStaticVal
	params[defaultFsNameKey] = defaultFSNameVal
	params[nodeStageSecretNamespaceKey] = defaultAdminSecretNamespace
}

// composeMigVolID composes migration handle for CephFS PV
// mig_mons-afcca55bc1bdd3f479be1e8281c13ab1_path-e0b45b52-7e09-47d3-8f1b-806995fa4412_7265706c696361706f6f6c
func composeMigVolIDForCephFs(mons string, path string) string {
	clusterIDInHandle := md5.Sum([]byte(mons))
	clusterField := monsPfx + fmt.Sprintf("%x", clusterIDInHandle)
	pathHashInHandle := strings.Split(path, defaultIntreeImagePfx)[1]
	pathField := cephfspathPfx + pathHashInHandle
	volHash := strings.Join([]string{migVolPfx, clusterField, pathField}, "_")
	return volHash
}

// fillVolAttrsForCephFSRequest fill the volume attributes for node operations
func fillVolAttrsForCephFSRequest(pv *v1.PersistentVolume, volumeAttributes map[string]string) error {
	if pv == nil || pv.Spec.CephFS == nil {
		return fmt.Errorf("pv is nil or CephFS Volume not defined on pv")
	}
	volumeAttributes[defaultRootPathKey] = pv.Spec.CephFS.Path
	volumeAttributes[defaultMigKey] = defaultMigStaticVal
	return nil
}

// fillAnnotationsFromCephFSCSISource capture required information from csi source
func fillAnnotationsFromCephFSCSISource(pv *v1.PersistentVolume, csiSource *v1.CSIPersistentVolumeSource) {
	pv.Annotations[CSICephFSVolHandleAnnKey] = csiSource.VolumeHandle
	pv.Annotations[clusterIDKey] = csiSource.VolumeAttributes[clusterIDKey]
}
