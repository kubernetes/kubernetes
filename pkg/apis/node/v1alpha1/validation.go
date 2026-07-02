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

package v1alpha1

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// MaxCheckpointedPodTemplateBytes bounds the protobuf encoding of
// status.checkpointedPodTemplate. Every get, list and watch of the
// PodCheckpoint carries the template, and the template cannot be replaced once
// written. The value must match the limit in the field's API documentation.
// One MiB follows the payload limits used by Secrets, ConfigMaps and
// ClusterTrustBundles while leaving room for the rest of the checkpoint object.
const MaxCheckpointedPodTemplateBytes = 1024 * 1024

// ValidateCustom_PodCheckpointStatus_CheckpointedPodTemplate is wired into the
// generated declarative validation by +k8s:customValidation on
// nodev1alpha1.PodCheckpointStatus.CheckpointedPodTemplate. The protobuf size
// bounds the complete captured template rather than an individual field.
func ValidateCustom_PodCheckpointStatus_CheckpointedPodTemplate(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *corev1.PodTemplateSpec) field.ErrorList {
	if value == nil {
		return nil
	}
	if oldValue != nil && equality.Semantic.DeepEqual(value, oldValue) {
		return nil
	}
	if value.Size() > MaxCheckpointedPodTemplateBytes {
		return field.ErrorList{field.TooLong(fldPath, "", MaxCheckpointedPodTemplateBytes)}
	}
	return nil
}
