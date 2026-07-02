/*
Copyright The Kubernetes Authors.

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
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	checkpoint "k8s.io/kubernetes/pkg/apis/node"
)

// ValidatePodCheckpointName validates the name of a PodCheckpoint.
func ValidatePodCheckpointName(name string, prefix bool) []string {
	return apimachineryvalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidatePodCheckpoint validates a PodCheckpoint.
func ValidatePodCheckpoint(pc *checkpoint.PodCheckpoint) field.ErrorList {
	// Spec validation is declarative, including immutability on update.
	return apimachineryvalidation.ValidateObjectMeta(&pc.ObjectMeta, true, ValidatePodCheckpointName, field.NewPath("metadata"))
}

// ValidatePodCheckpointUpdate validates a PodCheckpoint update.
func ValidatePodCheckpointUpdate(newPC, oldPC *checkpoint.PodCheckpoint) field.ErrorList {
	return apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))
}

// ValidatePodCheckpointStatusUpdate validates a status update of a PodCheckpoint.
func ValidatePodCheckpointStatusUpdate(newPC, oldPC *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, metav1validation.ValidateConditions(newPC.Status.Conditions, field.NewPath("status", "conditions"))...)
	return allErrs
}
