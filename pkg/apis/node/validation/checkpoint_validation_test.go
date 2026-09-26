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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	checkpoint "k8s.io/kubernetes/pkg/apis/node"
)

func newPodCheckpoint() *checkpoint.PodCheckpoint {
	return &checkpoint.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "cp-1", Namespace: "ns"},
		Spec: checkpoint.PodCheckpointSpec{
			SourcePod: &checkpoint.PodReference{Name: "my-app"},
		},
	}
}

func TestValidatePodCheckpointStatusUpdate(t *testing.T) {
	old := newPodCheckpoint()
	old.ResourceVersion = "1"

	valid := old.DeepCopy()
	valid.Status.Conditions = []metav1.Condition{{
		Type:               checkpoint.PodCheckpointConditionReady,
		Status:             metav1.ConditionTrue,
		Reason:             checkpoint.PodCheckpointReasonCompleted,
		LastTransitionTime: metav1.Now(),
	}}
	if errs := ValidatePodCheckpointStatusUpdate(valid, old); len(errs) != 0 {
		t.Errorf("expected no errors for valid status, got %v", errs)
	}

	bad := old.DeepCopy()
	bad.Status.Conditions = []metav1.Condition{{
		Type:   checkpoint.PodCheckpointConditionReady,
		Status: "Maybe", // not a valid ConditionStatus
		Reason: checkpoint.PodCheckpointReasonCompleted,
	}}
	if errs := ValidatePodCheckpointStatusUpdate(bad, old); len(errs) == 0 {
		t.Errorf("expected errors for invalid condition, got none")
	}
}
