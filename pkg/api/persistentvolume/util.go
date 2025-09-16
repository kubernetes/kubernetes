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

package persistentvolume

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	nodeapi "k8s.io/kubernetes/pkg/api/node"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledSpecFields removes disabled fields from the pv spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pv spec.
func DropDisabledSpecFields(pvSpec *api.PersistentVolumeSpec, oldPVSpec *api.PersistentVolumeSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeAttributesClass) {
		if oldPVSpec == nil || oldPVSpec.VolumeAttributesClassName == nil {
			pvSpec.VolumeAttributesClassName = nil
		}
	}
}

func GetWarningsForPersistentVolume(pv *api.PersistentVolume) []string {
	if pv == nil {
		return nil
	}
	return warningsForPersistentVolumeSpecAndMeta(nil, &pv.Spec, &pv.ObjectMeta)
}

var deprecatedAnnotations = []struct {
	key     string
	message string
}{
	{
		key:     `volume.beta.kubernetes.io/storage-class`,
		message: `deprecated since v1.8; use "storageClassName" attribute instead`,
	},
	{
		key:     `volume.beta.kubernetes.io/mount-options`,
		message: `deprecated since v1.31; use "mountOptions" attribute instead`,
	},
}

func warningsForPersistentVolumeSpecAndMeta(fieldPath *field.Path, pvSpec *api.PersistentVolumeSpec, pvMeta *metav1.ObjectMeta) []string {
	var warnings []string

	// use of deprecated annotations
	for _, deprecated := range deprecatedAnnotations {
		if _, exists := pvMeta.Annotations[deprecated.key]; exists {
			warnings = append(warnings, fmt.Sprintf("%s: %s", fieldPath.Child("metadata", "annotations").Key(deprecated.key), deprecated.message))
		}
	}

	if pvSpec.PersistentVolumeReclaimPolicy == api.PersistentVolumeReclaimRecycle {
		warnings = append(warnings, fmt.Sprintf("%s: The Recycle reclaim policy is deprecated. Instead, the recommended approach is to use dynamic provisioning.", fieldPath.Child("spec", "persistentVolumeReclaimPolicy")))
	}

	if pvSpec.NodeAffinity != nil && pvSpec.NodeAffinity.Required != nil {
		termFldPath := fieldPath.Child("spec", "nodeAffinity", "required", "nodeSelectorTerms")
		// use of deprecated node labels in node affinity
		for i, term := range pvSpec.NodeAffinity.Required.NodeSelectorTerms {
			warnings = append(warnings, nodeapi.GetWarningsForNodeSelectorTerm(term, false, termFldPath.Index(i))...)
		}
	}
	// If we are on deprecated volume plugin
	if pvSpec.CephFS != nil {
		warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.28, non-functional in v1.31+", fieldPath.Child("spec", "cephfs")))
	}
	if pvSpec.PhotonPersistentDisk != nil {
		warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.11, non-functional in v1.16+", fieldPath.Child("spec", "photonPersistentDisk")))
	}
	if pvSpec.ScaleIO != nil {
		warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.16, non-functional in v1.22+", fieldPath.Child("spec", "scaleIO")))
	}
	if pvSpec.StorageOS != nil {
		warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.22, non-functional in v1.25+", fieldPath.Child("spec", "storageOS")))
	}
	if pvSpec.Glusterfs != nil {
		warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.25, non-functional in v1.26+", fieldPath.Child("spec", "glusterfs")))
	}
	if pvSpec.RBD != nil {
		warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.28, non-functional in v1.31+", fieldPath.Child("spec", "rbd")))
	}
	return warnings
}
