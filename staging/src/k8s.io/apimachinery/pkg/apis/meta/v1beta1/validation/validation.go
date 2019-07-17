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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateTableOptions returns any invalid flags on TableOptions.
func ValidateTableOptions(opts *metav1beta1.TableOptions) field.ErrorList {
	var allErrs field.ErrorList
	switch opts.IncludeObject {
	case metav1.IncludeMetadata, metav1.IncludeNone, metav1.IncludeObject, "":
	default:
		allErrs = append(allErrs, field.Invalid(field.NewPath("includeObject"), opts.IncludeObject, "must be 'Metadata', 'Object', 'None', or empty"))
	}
	return allErrs
}
