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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
)

// ValidatePersistentVolume validates PV object for plugin specific validation
// We can put here validations which are specific to volume types.
func ValidatePersistentVolume(pv *api.PersistentVolume) field.ErrorList {
	allErrs := field.ErrorList{}
	if pv.Spec.HostPath != nil {
		return validateHostPathPV(pv)
	} else if pv.Spec.FlexVolume != nil {
		return validateFlexVolume(pv)
	} else if pv.Spec.FC != nil {
		return validateFCVolume(pv)
	} else if pv.Spec.Flocker != nil {
		return validateFlockerVolume(pv)
	} else if pv.Spec.PortworxVolume != nil {
		return validatePortWorxVolume(pv)
	} else if pv.Spec.ScaleIO != nil {
		return validateScaleIOVolume(pv)
	}
	return allErrs
}

func validateHostPathPV(pv *api.PersistentVolume) field.ErrorList {
	return checkMountOption(pv)
}

func validateFlexVolume(pv *api.PersistentVolume) field.ErrorList {
	return checkMountOption(pv)
}

func validateFCVolume(pv *api.PersistentVolume) field.ErrorList {
	return checkMountOption(pv)
}

func validateFlockerVolume(pv *api.PersistentVolume) field.ErrorList {
	return checkMountOption(pv)
}

func validatePortWorxVolume(pv *api.PersistentVolume) field.ErrorList {
	return checkMountOption(pv)
}

func validateScaleIOVolume(pv *api.PersistentVolume) field.ErrorList {
	return checkMountOption(pv)
}

func checkMountOption(pv *api.PersistentVolume) field.ErrorList {
	allErrs := field.ErrorList{}
	mountOptions := volume.MountOptionFromApiPV(pv)
	metaField := field.NewPath("metadata")

	if len(mountOptions) > 0 {
		allErrs = append(allErrs, field.Forbidden(metaField.Child("annotations", volume.MountOptionAnnotation), "may not specify mount options for this volume type"))
	}
	return allErrs
}
