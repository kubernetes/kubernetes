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
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestCheckpointedPodTemplateSize(t *testing.T) {
	large := &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"large": strings.Repeat("x", MaxCheckpointedPodTemplateBytes)}}}
	if errs := ValidateCustom_PodCheckpointStatus_CheckpointedPodTemplate(context.Background(), operation.Operation{}, field.NewPath("status", "checkpointedPodTemplate"), large, nil); len(errs) == 0 {
		t.Fatal("expected oversized initial template to be rejected")
	}
	if errs := ValidateCustom_PodCheckpointStatus_CheckpointedPodTemplate(context.Background(), operation.Operation{}, field.NewPath("status", "checkpointedPodTemplate"), large, large.DeepCopy()); len(errs) != 0 {
		t.Fatalf("unchanged oversized stored template rejected: %v", errs)
	}
}
