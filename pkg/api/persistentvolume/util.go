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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// DropDisabledFields removes disabled fields from the pv spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pv spec.
func DropDisabledFields(pvSpec *api.PersistentVolumeSpec, oldPVSpec *api.PersistentVolumeSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandCSIVolumes) && !hasExpansionSecrets(oldPVSpec) {
		if pvSpec.CSI != nil {
			pvSpec.CSI.ControllerExpandSecretRef = nil
		}
	}
}

// AddDisabledFieldsTo to the provided set of resetFields
// This should be called with the resetFields for all resources containing a Pod
func AddDisabledFieldsTo(resetFields *fieldpath.Set) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandCSIVolumes) {
		// TODO: make sure this path is correct, the api type has some inlines
		resetFields.Insert(fieldpath.MakePathOrDie("spec", "csi", "controllerExpandSecretRef"))
	}
}

func hasExpansionSecrets(oldPVSpec *api.PersistentVolumeSpec) bool {
	if oldPVSpec == nil || oldPVSpec.CSI == nil {
		return false
	}

	if oldPVSpec.CSI.ControllerExpandSecretRef != nil {
		return true
	}
	return false
}
