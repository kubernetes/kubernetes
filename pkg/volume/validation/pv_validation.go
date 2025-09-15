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

package validation

import (
	"errors"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// ValidatePersistentVolume validates PV object for plugin specific validation
// We can put here validations which are specific to volume types.
func ValidatePersistentVolume(pv *api.PersistentVolume) field.ErrorList {
	return checkMountOption(pv)
}

func checkMountOption(pv *api.PersistentVolume) field.ErrorList {
	allErrs := field.ErrorList{}
	// if PV is of these types we don't return errors
	// since mount options is supported
	if pv.Spec.GCEPersistentDisk != nil ||
		pv.Spec.AWSElasticBlockStore != nil ||
		pv.Spec.Glusterfs != nil ||
		pv.Spec.NFS != nil ||
		pv.Spec.RBD != nil ||
		pv.Spec.Quobyte != nil ||
		pv.Spec.ISCSI != nil ||
		pv.Spec.Cinder != nil ||
		pv.Spec.CephFS != nil ||
		pv.Spec.AzureFile != nil ||
		pv.Spec.VsphereVolume != nil ||
		pv.Spec.AzureDisk != nil ||
		pv.Spec.PhotonPersistentDisk != nil {
		return allErrs
	}
	// any other type if mount option is present lets return error
	if _, ok := pv.Annotations[api.MountOptionAnnotation]; ok {
		metaField := field.NewPath("metadata")
		allErrs = append(allErrs, field.Forbidden(metaField.Child("annotations", api.MountOptionAnnotation), "may not specify mount options for this volume type"))
	}
	return allErrs
}

// ValidatePathNoBacksteps will make sure the targetPath does not have any element which is ".."
func ValidatePathNoBacksteps(targetPath string) error {
	parts := strings.Split(filepath.ToSlash(targetPath), "/")
	for _, item := range parts {
		if item == ".." {
			return errors.New("must not contain '..'")
		}
	}

	return nil
}
