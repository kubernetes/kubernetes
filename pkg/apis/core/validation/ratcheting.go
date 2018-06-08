/*
Copyright 2018 The Kubernetes Authors.

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
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
)

var validStorageMedia = sets.NewString(
	string(core.StorageMediumDefault),
	string(core.StorageMediumMemory),
	string(core.StorageMediumHugePages),
)

// RatchetingPodSpecValidations is a set of additional validations added to validate pod specs.
// This should be passed to ValidateRatchetingCreate and ValidateRatchetingUpdate along with *core.PodSpec data to validate.
var RatchetingPodSpecValidations = apimachineryvalidation.RatchetingValidators{
	// flex volume source driver name
	func(data interface{}, fldPath *field.Path) field.ErrorList {
		allErrs := field.ErrorList{}
		spec := data.(*core.PodSpec)
		for i, v := range spec.Volumes {
			if v.VolumeSource.FlexVolume != nil && len(v.VolumeSource.FlexVolume.Driver) > 0 {
				for _, msg := range validation.IsQualifiedName(v.VolumeSource.FlexVolume.Driver) {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("volumes").Index(i).Child("flexVolume", "driver"), v.VolumeSource.FlexVolume.Driver, msg))
				}
			}
		}
		return allErrs
	},

	// emptyDir storage medium
	func(data interface{}, fldPath *field.Path) field.ErrorList {
		allErrs := field.ErrorList{}
		spec := data.(*core.PodSpec)
		for i, v := range spec.Volumes {
			if v.VolumeSource.EmptyDir != nil {
				if !validStorageMedia.Has(string(v.VolumeSource.EmptyDir.Medium)) {
					allErrs = append(allErrs, field.NotSupported(fldPath.Child("volumes").Index(i).Child("emptyDir", "medium"), v.VolumeSource.EmptyDir.Medium, validStorageMedia.List()))
				}
			}
		}
		return allErrs
	},

	// duplicate envvar names
	func(data interface{}, fldPath *field.Path) field.ErrorList {
		allErrs := field.ErrorList{}
		visitContainers(data.(*core.PodSpec), fldPath, func(c *core.Container, fldPath *field.Path) {
			names := sets.NewString()
			for j, e := range c.Env {
				if names.Has(e.Name) {
					allErrs = append(allErrs, field.Duplicate(fldPath.Child("env").Index(j).Child("name"), e.Name))
				} else {
					names.Insert(e.Name)
				}
			}
		})
		return allErrs
	},

	// duplicate pvc volumes
	func(data interface{}, fldPath *field.Path) field.ErrorList {
		allErrs := field.ErrorList{}
		spec := data.(*core.PodSpec)
		names := sets.NewString()
		for i, v := range spec.Volumes {
			if pvc := v.VolumeSource.PersistentVolumeClaim; pvc != nil {
				if names.Has(pvc.ClaimName) {
					allErrs = append(allErrs, field.Duplicate(fldPath.Child("volumes").Index(i).Child("persistentVolumeClaim").Child("claimName"), pvc.ClaimName))
				} else {
					names.Insert(pvc.ClaimName)
				}
			}
		}
		return allErrs
	},

	// memory units
	func(data interface{}, fldPath *field.Path) field.ErrorList {
		allErrs := field.ErrorList{}
		visitContainers(data.(*core.PodSpec), fldPath, func(c *core.Container, fldPath *field.Path) {
			for k, q := range c.Resources.Limits {
				if k != core.ResourceMemory {
					continue
				}
				if _, isInteger := q.AsScale(0); !isInteger {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("resources", "limits", string(k)), q.String(), "must be an integer value"))
				}
			}
			for k, q := range c.Resources.Requests {
				if k != core.ResourceMemory {
					continue
				}
				if _, isInteger := q.AsScale(0); !isInteger {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("resources", "requests", string(k)), q.String(), "must be an integer value"))
				}
			}
		})

		return allErrs
	},
}

func visitContainers(podSpec *core.PodSpec, fldPath *field.Path, visitor func(*core.Container, *field.Path)) {
	for i, c := range podSpec.InitContainers {
		visitor(&c, fldPath.Child("initContainers").Index(i))
	}
	for i, c := range podSpec.Containers {
		visitor(&c, fldPath.Child("containers").Index(i))
	}
}
