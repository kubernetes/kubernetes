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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/storage"
)

// ValidateStorageClass validates a StorageClass.
func ValidateStorageClass(storageClass *storage.StorageClass) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&storageClass.ObjectMeta, false, apivalidation.ValidateClassName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateProvisioner(storageClass.Provisioner, field.NewPath("provisioner"))...)
	allErrs = append(allErrs, validateParameters(storageClass.Parameters, field.NewPath("parameters"))...)
	allErrs = append(allErrs, validateReclaimPolicy(storageClass.ReclaimPolicy, field.NewPath("reclaimPolicy"))...)

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

const maxProvisionerParameterSize = 256 * (1 << 10) // 256 kB
const maxProvisionerParameterLen = 512

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
