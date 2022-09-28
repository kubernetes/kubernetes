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
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	nodeapi "k8s.io/kubernetes/pkg/api/node"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledFields removes disabled fields from the pv spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pv spec.
func DropDisabledFields(pvSpec *api.PersistentVolumeSpec, oldPVSpec *api.PersistentVolumeSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSINodeExpandSecret) && !hasNodeExpansionSecrets(oldPVSpec) {
		if pvSpec.CSI != nil {
			pvSpec.CSI.NodeExpandSecretRef = nil
		}
	}
}

func hasNodeExpansionSecrets(oldPVSpec *api.PersistentVolumeSpec) bool {
	if oldPVSpec == nil || oldPVSpec.CSI == nil {
		return false
	}

	if oldPVSpec.CSI.NodeExpandSecretRef != nil {
		return true
	}
	return false
}

func GetWarningsForPersistentVolume(pv *api.PersistentVolume) []string {
	if pv == nil {
		return nil
	}
	return warningsForPersistentVolumeSpecAndMeta(nil, &pv.Spec)
}

func warningsForPersistentVolumeSpecAndMeta(fieldPath *field.Path, pvSpec *api.PersistentVolumeSpec) []string {
	var warnings []string

	if pvSpec.NodeAffinity != nil && pvSpec.NodeAffinity.Required != nil {
		termFldPath := fieldPath.Child("spec", "nodeAffinity", "required", "nodeSelectorTerms")
		// use of deprecated node labels in node affinity
		for i, term := range pvSpec.NodeAffinity.Required.NodeSelectorTerms {
			warnings = append(warnings, nodeapi.GetWarningsForNodeSelectorTerm(term, termFldPath.Index(i))...)
		}
	}

	return warnings
}
