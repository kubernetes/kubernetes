/*
Copyright 2015 The Kubernetes Authors.

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

package validation

import (
	"reflect"
	"strings"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
)

const (
	maxProvisionerParameterSize = 256 * (1 << 10) // 256 kB
	maxProvisionerParameterLen  = 512

	maxAttachedVolumeMetadataSize = 256 * (1 << 10) // 256 kB
	maxVolumeErrorMessageSize     = 1024
)

// ValidateStorageClass validates a StorageClass.
func ValidateStorageClass(storageClass *storage.StorageClass) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&storageClass.ObjectMeta, false, apivalidation.ValidateClassName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateProvisioner(storageClass.Provisioner, field.NewPath("provisioner"))...)
	allErrs = append(allErrs, validateParameters(storageClass.Parameters, field.NewPath("parameters"))...)
	allErrs = append(allErrs, validateReclaimPolicy(storageClass.ReclaimPolicy, field.NewPath("reclaimPolicy"))...)
	allErrs = append(allErrs, validateAllowVolumeExpansion(storageClass.AllowVolumeExpansion, field.NewPath("allowVolumeExpansion"))...)
	allErrs = append(allErrs, validateVolumeBindingMode(storageClass.VolumeBindingMode, field.NewPath("volumeBindingMode"))...)
	allErrs = append(allErrs, validateAllowedTopologies(storageClass.AllowedTopologies, field.NewPath("allowedTopologies"))...)

	return allErrs
}

// ValidateStorageClassUpdate tests if an update to StorageClass is valid.
func ValidateStorageClassUpdate(storageClass, oldStorageClass *storage.StorageClass) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&storageClass.ObjectMeta, &oldStorageClass.ObjectMeta, field.NewPath("metadata"))
	if !reflect.DeepEqual(oldStorageClass.Parameters, storageClass.Parameters) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("parameters"), "updates to parameters are forbidden."))
	}

	if storageClass.Provisioner != oldStorageClass.Provisioner {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("provisioner"), "updates to provisioner are forbidden."))
	}

	if *storageClass.ReclaimPolicy != *oldStorageClass.ReclaimPolicy {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("reclaimPolicy"), "updates to reclaimPolicy are forbidden."))
	}

	allErrs = append(allErrs, apivalidation.ValidateImmutableField(storageClass.VolumeBindingMode, oldStorageClass.VolumeBindingMode, field.NewPath("volumeBindingMode"))...)
	return allErrs
}

// validateProvisioner tests if provisioner is a valid qualified name.
func validateProvisioner(provisioner string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(provisioner) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, provisioner))
	}
	if len(provisioner) > 0 {
		for _, msg := range validation.IsQualifiedName(strings.ToLower(provisioner)) {
			allErrs = append(allErrs, field.Invalid(fldPath, provisioner, msg))
		}
	}
	return allErrs
}

// validateParameters tests that keys are qualified names and that provisionerParameter are < 256kB.
func validateParameters(params map[string]string, fldPath *field.Path) field.ErrorList {
	var totalSize int64
	allErrs := field.ErrorList{}

	if len(params) > maxProvisionerParameterLen {
		allErrs = append(allErrs, field.TooLong(fldPath, "Provisioner Parameters exceeded max allowed", maxProvisionerParameterLen))
		return allErrs
	}

	for k, v := range params {
		if len(k) < 1 {
			allErrs = append(allErrs, field.Invalid(fldPath, k, "field can not be empty."))
		}
		totalSize += (int64)(len(k)) + (int64)(len(v))
	}

	if totalSize > maxProvisionerParameterSize {
		allErrs = append(allErrs, field.TooLong(fldPath, "", maxProvisionerParameterSize))
	}
	return allErrs
}

var supportedReclaimPolicy = sets.NewString(string(api.PersistentVolumeReclaimDelete), string(api.PersistentVolumeReclaimRetain))

// validateReclaimPolicy tests that the reclaim policy is one of the supported. It is up to the volume plugin to reject
// provisioning for storage classes with impossible reclaim policies, e.g. EBS is not Recyclable
func validateReclaimPolicy(reclaimPolicy *api.PersistentVolumeReclaimPolicy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(string(*reclaimPolicy)) > 0 {
		if !supportedReclaimPolicy.Has(string(*reclaimPolicy)) {
			allErrs = append(allErrs, field.NotSupported(fldPath, reclaimPolicy, supportedReclaimPolicy.List()))
		}
	}
	return allErrs
}

// validateAllowVolumeExpansion tests that if ExpandPersistentVolumes feature gate is disabled, whether the AllowVolumeExpansion filed
// of storage class is set
func validateAllowVolumeExpansion(allowExpand *bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if allowExpand != nil && !utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		allErrs = append(allErrs, field.Forbidden(fldPath, "field is disabled by feature-gate ExpandPersistentVolumes"))
	}
	return allErrs
}

// ValidateVolumeAttachment validates a VolumeAttachment.
func ValidateVolumeAttachment(volumeAttachment *storage.VolumeAttachment) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&volumeAttachment.ObjectMeta, false, apivalidation.ValidateClassName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateVolumeAttachmentSpec(&volumeAttachment.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, validateVolumeAttachmentStatus(&volumeAttachment.Status, field.NewPath("status"))...)
	return allErrs
}

// ValidateVolumeAttachmentSpec tests that the specified VolumeAttachmentSpec
// has valid data.
func validateVolumeAttachmentSpec(
	spec *storage.VolumeAttachmentSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validateAttacher(spec.Attacher, fldPath.Child("attacher"))...)
	allErrs = append(allErrs, validateVolumeAttachmentSource(&spec.Source, fldPath.Child("source"))...)
	allErrs = append(allErrs, validateNodeName(spec.NodeName, fldPath.Child("nodeName"))...)
	return allErrs
}

// validateAttacher tests if attacher is a valid qualified name.
func validateAttacher(attacher string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(attacher) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, attacher))
	}
	return allErrs
}

// validateSource tests if the source is valid for VolumeAttachment.
func validateVolumeAttachmentSource(source *storage.VolumeAttachmentSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if source.PersistentVolumeName == nil || len(*source.PersistentVolumeName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	}
	return allErrs
}

// validateNodeName tests if the nodeName is valid for VolumeAttachment.
func validateNodeName(nodeName string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range apivalidation.ValidateNodeName(nodeName, false /* prefix */) {
		allErrs = append(allErrs, field.Invalid(fldPath, nodeName, msg))
	}
	return allErrs
}

// validaVolumeAttachmentStatus tests if volumeAttachmentStatus is valid.
func validateVolumeAttachmentStatus(status *storage.VolumeAttachmentStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validateAttachmentMetadata(status.AttachmentMetadata, fldPath.Child("attachmentMetadata"))...)
	allErrs = append(allErrs, validateVolumeError(status.AttachError, fldPath.Child("attachError"))...)
	allErrs = append(allErrs, validateVolumeError(status.DetachError, fldPath.Child("detachError"))...)
	return allErrs
}

func validateAttachmentMetadata(metadata map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	var size int64
	for k, v := range metadata {
		size += (int64)(len(k)) + (int64)(len(v))
	}
	if size > maxAttachedVolumeMetadataSize {
		allErrs = append(allErrs, field.TooLong(fldPath, metadata, maxAttachedVolumeMetadataSize))
	}
	return allErrs
}

func validateVolumeError(e *storage.VolumeError, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if e == nil {
		return allErrs
	}
	if len(e.Message) > maxVolumeErrorMessageSize {
		allErrs = append(allErrs, field.TooLong(fldPath.Child("message"), e.Message, maxAttachedVolumeMetadataSize))
	}
	return allErrs
}

// ValidateVolumeAttachmentUpdate validates a VolumeAttachment.
func ValidateVolumeAttachmentUpdate(new, old *storage.VolumeAttachment) field.ErrorList {
	allErrs := ValidateVolumeAttachment(new)

	// Spec is read-only
	if !apiequality.Semantic.DeepEqual(old.Spec, new.Spec) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec"), new.Spec, "field is immutable"))
	}
	return allErrs
}

var supportedVolumeBindingModes = sets.NewString(string(storage.VolumeBindingImmediate), string(storage.VolumeBindingWaitForFirstConsumer))

// validateVolumeBindingMode tests that VolumeBindingMode specifies valid values.
func validateVolumeBindingMode(mode *storage.VolumeBindingMode, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		if mode == nil {
			allErrs = append(allErrs, field.Required(fldPath, ""))
		} else if !supportedVolumeBindingModes.Has(string(*mode)) {
			allErrs = append(allErrs, field.NotSupported(fldPath, mode, supportedVolumeBindingModes.List()))
		}
	} else if mode != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath, "field is disabled by feature-gate VolumeScheduling"))
	}

	return allErrs
}

// validateAllowedTopology tests that AllowedTopologies specifies valid values.
func validateAllowedTopologies(topologies []api.TopologySelectorTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if topologies == nil || len(topologies) == 0 {
		return allErrs
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		allErrs = append(allErrs, field.Forbidden(fldPath, "field is disabled by feature-gate VolumeScheduling"))
	}

	rawTopologies := make([]map[string]sets.String, len(topologies))
	for i, term := range topologies {
		idxPath := fldPath.Index(i)
		exprMap, termErrs := apivalidation.ValidateTopologySelectorTerm(term, fldPath.Index(i))
		allErrs = append(allErrs, termErrs...)

		// TODO (verult) consider improving runtime
		for _, t := range rawTopologies {
			if helper.Semantic.DeepEqual(exprMap, t) {
				allErrs = append(allErrs, field.Duplicate(idxPath.Child("matchLabelExpressions"), ""))
			}
		}

		rawTopologies = append(rawTopologies, exprMap)
	}

	return allErrs
}
