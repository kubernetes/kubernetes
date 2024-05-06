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

package persistentvolumeclaim

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"
)

const (
	pvc                                  string = "PersistentVolumeClaim"
	volumeSnapshot                       string = "VolumeSnapshot"
	deprecatedStorageClassAnnotationsMsg        = `deprecated since v1.8; use "storageClassName" attribute instead`
)

// DropDisabledFields removes disabled fields from the pvc spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pvc spec.
func DropDisabledFields(pvcSpec, oldPVCSpec *core.PersistentVolumeClaimSpec) {
	// Drop the contents of the volumeAttributesClassName if the VolumeAttributesClass
	// feature gate is disabled.
	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeAttributesClass) {
		if oldPVCSpec == nil || oldPVCSpec.VolumeAttributesClassName == nil {
			pvcSpec.VolumeAttributesClassName = nil
		}
	}

	// Drop the contents of the dataSourceRef field if the AnyVolumeDataSource
	// feature gate is disabled.
	if !utilfeature.DefaultFeatureGate.Enabled(features.AnyVolumeDataSource) {
		if !dataSourceRefInUse(oldPVCSpec) {
			pvcSpec.DataSourceRef = nil
		}
	}

	// Drop the contents of the dataSourceRef field if the CrossNamespaceVolumeDataSource
	// feature gate is disabled and dataSourceRef.Namespace is specified.
	if !utilfeature.DefaultFeatureGate.Enabled(features.CrossNamespaceVolumeDataSource) &&
		pvcSpec.DataSourceRef != nil && pvcSpec.DataSourceRef.Namespace != nil && len(*pvcSpec.DataSourceRef.Namespace) != 0 {
		if !dataSourceRefInUse(oldPVCSpec) {
			pvcSpec.DataSourceRef = nil
		}
	}
}

// EnforceDataSourceBackwardsCompatibility drops the data source field under certain conditions
// to maintain backwards compatibility with old behavior.
// See KEP 1495 for details.
// Specifically, if this is an update of a PVC with no data source, or a creation of a new PVC,
// and the dataSourceRef field is not filled in, then we will drop "invalid" data sources
// (anything other than a PVC or a VolumeSnapshot) from this request as if an empty PVC had
// been requested.
// This should be called after DropDisabledFields so that if the AnyVolumeDataSource feature
// gate is disabled, dataSourceRef will be forced to empty, ensuring pre-1.22 behavior.
// This should be called before NormalizeDataSources, so that data sources other than PVCs
// and VolumeSnapshots can only be set through the dataSourceRef field and not the dataSource
// field.
func EnforceDataSourceBackwardsCompatibility(pvcSpec, oldPVCSpec *core.PersistentVolumeClaimSpec) {
	// Check if the old PVC has a data source here is so that on updates from old clients
	// that omit dataSourceRef, we preserve the data source, even if it would have been
	// invalid to specify it at using the dataSource field at create.
	if dataSourceInUse(oldPVCSpec) {
		return
	}

	// Check if dataSourceRef is empty is because if it's not empty, then there is
	// definitely a newer client and it definitely either wants to create a non-empty
	// volume, or it wants to update a PVC that has a data source. Whether the
	// specified data source is valid or satisfiable is a matter for validation and
	// the volume populator code, but we can say with certainty that the client is
	// not expecting the legacy behavior of ignoring invalid data sources.
	if pvcSpec.DataSourceRef != nil {
		return
	}

	// Historically, we only allow PVCs and VolumeSnapshots in the dataSource field.
	// All other values are silently dropped.
	if !dataSourceIsPvcOrSnapshot(pvcSpec.DataSource) {
		pvcSpec.DataSource = nil
	}
}

func DropDisabledFieldsFromStatus(pvc, oldPVC *core.PersistentVolumeClaim) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeAttributesClass) {
		if oldPVC == nil || oldPVC.Status.CurrentVolumeAttributesClassName == nil {
			pvc.Status.CurrentVolumeAttributesClassName = nil
		}
		if oldPVC == nil || oldPVC.Status.ModifyVolumeStatus == nil {
			pvc.Status.ModifyVolumeStatus = nil
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.RecoverVolumeExpansionFailure) {
		if !helper.ClaimContainsAllocatedResources(oldPVC) {
			pvc.Status.AllocatedResources = nil
		}
		if !helper.ClaimContainsAllocatedResourceStatus(oldPVC) {
			pvc.Status.AllocatedResourceStatuses = nil
		}
	}
}

func dataSourceInUse(oldPVCSpec *core.PersistentVolumeClaimSpec) bool {
	if oldPVCSpec == nil {
		return false
	}
	if oldPVCSpec.DataSource != nil || oldPVCSpec.DataSourceRef != nil {
		return true
	}
	return false
}

func dataSourceIsPvcOrSnapshot(dataSource *core.TypedLocalObjectReference) bool {
	if dataSource != nil {
		apiGroup := ""
		if dataSource.APIGroup != nil {
			apiGroup = *dataSource.APIGroup
		}
		if dataSource.Kind == pvc &&
			apiGroup == "" {
			return true
		}

		if dataSource.Kind == volumeSnapshot && apiGroup == "snapshot.storage.k8s.io" {
			return true
		}
	}
	return false
}

func dataSourceRefInUse(oldPVCSpec *core.PersistentVolumeClaimSpec) bool {
	if oldPVCSpec == nil {
		return false
	}
	if oldPVCSpec.DataSourceRef != nil {
		return true
	}
	return false
}

// NormalizeDataSources ensures that DataSource and DataSourceRef have the same contents
// as long as both are not explicitly set.
// This should be used by creates/gets of PVCs, but not updates
func NormalizeDataSources(pvcSpec *core.PersistentVolumeClaimSpec) {
	// Don't enable this behavior if the feature gate is not on
	if !utilfeature.DefaultFeatureGate.Enabled(features.AnyVolumeDataSource) {
		return
	}
	if pvcSpec.DataSource != nil && pvcSpec.DataSourceRef == nil {
		// Using the old way of setting a data source
		pvcSpec.DataSourceRef = &core.TypedObjectReference{
			Kind: pvcSpec.DataSource.Kind,
			Name: pvcSpec.DataSource.Name,
		}
		if pvcSpec.DataSource.APIGroup != nil {
			apiGroup := *pvcSpec.DataSource.APIGroup
			pvcSpec.DataSourceRef.APIGroup = &apiGroup
		}
	} else if pvcSpec.DataSourceRef != nil && pvcSpec.DataSource == nil {
		if pvcSpec.DataSourceRef.Namespace == nil || len(*pvcSpec.DataSourceRef.Namespace) == 0 {
			// Using the new way of setting a data source
			pvcSpec.DataSource = &core.TypedLocalObjectReference{
				Kind: pvcSpec.DataSourceRef.Kind,
				Name: pvcSpec.DataSourceRef.Name,
			}
			if pvcSpec.DataSourceRef.APIGroup != nil {
				apiGroup := *pvcSpec.DataSourceRef.APIGroup
				pvcSpec.DataSource.APIGroup = &apiGroup
			}
		}
	}
}

func GetWarningsForPersistentVolumeClaim(pv *core.PersistentVolumeClaim) []string {
	var warnings []string

	if pv == nil {
		return nil
	}

	if _, ok := pv.ObjectMeta.Annotations[core.BetaStorageClassAnnotation]; ok {
		warnings = append(warnings,
			fmt.Sprintf(
				"%s: %s",
				field.NewPath("metadata", "annotations").Key(core.BetaStorageClassAnnotation),
				deprecatedStorageClassAnnotationsMsg,
			),
		)
	}

	warnings = append(warnings, GetWarningsForPersistentVolumeClaimSpec(field.NewPath("spec"), pv.Spec)...)

	return warnings
}

func GetWarningsForPersistentVolumeClaimSpec(fieldPath *field.Path, pvSpec core.PersistentVolumeClaimSpec) []string {

	var warnings []string
	requestValue := pvSpec.Resources.Requests[core.ResourceStorage]
	if requestValue.MilliValue()%int64(1000) != int64(0) {
		warnings = append(warnings, fmt.Sprintf(
			"%s: fractional byte value %q is invalid, must be an integer",
			fieldPath.Child("resources").Child("requests").Key(core.ResourceStorage.String()), requestValue.String()))
	}
	limitValue := pvSpec.Resources.Limits[core.ResourceStorage]
	if limitValue.MilliValue()%int64(1000) != int64(0) {
		warnings = append(warnings, fmt.Sprintf(
			"%s: fractional byte value %q is invalid, must be an integer",
			fieldPath.Child("resources").Child("limits").Key(core.ResourceStorage.String()), limitValue.String()))
	}
	return warnings
}
