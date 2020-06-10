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
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const (
	pvc            string = "PersistentVolumeClaim"
	volumeSnapshot string = "VolumeSnapshot"
)

// DropDisabledFields removes disabled fields from the pvc spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pvc spec.
func DropDisabledFields(pvcSpec, oldPVCSpec *core.PersistentVolumeClaimSpec) {
	if !dataSourceIsEnabled(pvcSpec) && !dataSourceInUse(oldPVCSpec) {
		pvcSpec.DataSource = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.RecoverVolumeExpansionFailure) && !allocatedResourcesInUse(oldPVCSpec) {
		pvcSpec.AllocatedResources = nil
	}
}

// SetAllocatedResources ensures that AllocatedResources field can only increase and tracks
// maximum user requested capacity.
func SetAllocatedResources(pvcSpec, oldPVCSpec *core.PersistentVolumeClaimSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.RecoverVolumeExpansionFailure) {
		userResources := pvcSpec.Resources

		// for new PVC creation we will simply set it to user requested size
		// even if user provided one.
		if oldPVCSpec == nil {
			pvcSpec.AllocatedResources = &userResources
			return
		}

		var oldAllocatedQuantity *resource.Quantity

		if oldPVCSpec.AllocatedResources != nil {
			oldAllocatedQuantity = oldPVCSpec.AllocatedResources.Requests.Storage()
		}

		// We generally do not want users to supply their own value of allocated resources
		// but if old PVC did not have AllocatedResources set then we can use old user requested
		// resource.
		if oldAllocatedQuantity == nil {
			pvcSpec.AllocatedResources = &oldPVCSpec.Resources
		} else {
			// overwrite any user provided value of AllocatedResources and use old AllocatedResources
			pvcSpec.AllocatedResources = oldPVCSpec.AllocatedResources
		}

		allocatedResource := pvcSpec.AllocatedResources.Requests[core.ResourceStorage]
		userRequestedSize := userResources.Requests[core.ResourceStorage]

		if userRequestedSize.Cmp(allocatedResource) > 0 {
			pvcSpec.AllocatedResources = &userResources
		}
	}
}

func allocatedResourcesInUse(oldPVCSpec *core.PersistentVolumeClaimSpec) bool {
	if oldPVCSpec == nil {
		return false
	}

	if oldPVCSpec.AllocatedResources != nil {
		return true
	}

	return false
}

func dataSourceInUse(oldPVCSpec *core.PersistentVolumeClaimSpec) bool {
	if oldPVCSpec == nil {
		return false
	}
	if oldPVCSpec.DataSource != nil {
		return true
	}
	return false
}

func dataSourceIsEnabled(pvcSpec *core.PersistentVolumeClaimSpec) bool {
	if pvcSpec.DataSource != nil {
		if utilfeature.DefaultFeatureGate.Enabled(features.AnyVolumeDataSource) {
			return true
		}

		apiGroup := ""
		if pvcSpec.DataSource.APIGroup != nil {
			apiGroup = *pvcSpec.DataSource.APIGroup
		}
		if pvcSpec.DataSource.Kind == pvc &&
			apiGroup == "" {
			return true

		}

		if utilfeature.DefaultFeatureGate.Enabled(features.VolumeSnapshotDataSource) &&
			pvcSpec.DataSource.Kind == volumeSnapshot &&
			apiGroup == "snapshot.storage.k8s.io" {
			return true
		}
	}
	return false
}
