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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/apis/checkpoint"
	"k8s.io/utils/ptr"
)

func newPodCheckpoint() *checkpoint.PodCheckpoint {
	return &checkpoint.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "cp-1", Namespace: "ns"},
		Spec: checkpoint.PodCheckpointSpec{
			SourcePod: &checkpoint.PodReference{Name: "my-app"},
		},
	}
}

func TestValidatePodCheckpoint(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*checkpoint.PodCheckpoint)
		wantErr bool
	}{
		{name: "valid", mutate: func(*checkpoint.PodCheckpoint) {}},
		{name: "valid timeout", mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.TimeoutSeconds = ptr.To[int32](30)
		}},
		{name: "valid maximum timeout", mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.TimeoutSeconds = ptr.To[int32](MaxTimeoutSeconds)
		}},
		{name: "valid sourcePod.uid", mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.SourcePod.UID = ptr.To(types.UID("7b2c1e4a-0e3a-4f1b-9c2d-2a5f6e8d1234"))
		}},
		{name: "empty sourcePod.uid", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.SourcePod.UID = ptr.To(types.UID(""))
		}},
		{name: "missing sourcePod", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.SourcePod = nil
		}},
		{name: "missing sourcePod.name", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.SourcePod.Name = ""
		}},
		{name: "invalid sourcePod.name", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.SourcePod.Name = "Not_A_Valid_Name"
		}},
		{name: "negative timeout", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.TimeoutSeconds = ptr.To[int32](-1)
		}},
		{name: "zero timeout is not unset", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.TimeoutSeconds = ptr.To[int32](0)
		}},
		{name: "timeout above maximum", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.TimeoutSeconds = ptr.To[int32](MaxTimeoutSeconds + 1)
		}},
		{name: "missing name", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Name = ""
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pc := newPodCheckpoint()
			tc.mutate(pc)
			errs := ValidatePodCheckpoint(pc)
			if tc.wantErr && len(errs) == 0 {
				t.Errorf("expected errors, got none")
			}
			if !tc.wantErr && len(errs) != 0 {
				t.Errorf("expected no errors, got %v", errs)
			}
		})
	}
}

func TestValidatePodCheckpointUpdate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*checkpoint.PodCheckpoint)
		wantErr bool
	}{
		{name: "no change", mutate: func(*checkpoint.PodCheckpoint) {}},
		{name: "sourcePod.name immutable", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.SourcePod.Name = "other-pod"
		}},
		{name: "sourcePod.uid immutable", wantErr: true, mutate: func(pc *checkpoint.PodCheckpoint) {
			pc.Spec.SourcePod.UID = ptr.To(types.UID("7b2c1e4a-0e3a-4f1b-9c2d-2a5f6e8d1234"))
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			old := newPodCheckpoint()
			old.ResourceVersion = "1"
			newPC := old.DeepCopy()
			tc.mutate(newPC)
			errs := ValidatePodCheckpointUpdate(newPC, old)
			if tc.wantErr && len(errs) == 0 {
				t.Errorf("expected errors, got none")
			}
			if !tc.wantErr && len(errs) != 0 {
				t.Errorf("expected no errors, got %v", errs)
			}
		})
	}
}

func TestValidatePodCheckpointStatusUpdate(t *testing.T) {
	old := newPodCheckpoint()
	old.ResourceVersion = "1"

	valid := old.DeepCopy()
	valid.Status.Conditions = []metav1.Condition{{
		Type:               checkpoint.PodCheckpointReady,
		Status:             metav1.ConditionTrue,
		Reason:             checkpoint.PodCheckpointReasonCompleted,
		LastTransitionTime: metav1.Now(),
	}}
	if errs := ValidatePodCheckpointStatusUpdate(valid, old); len(errs) != 0 {
		t.Errorf("expected no errors for valid status, got %v", errs)
	}

	bad := old.DeepCopy()
	bad.Status.Conditions = []metav1.Condition{{
		Type:   checkpoint.PodCheckpointReady,
		Status: "Maybe", // not a valid ConditionStatus
		Reason: checkpoint.PodCheckpointReasonCompleted,
	}}
	if errs := ValidatePodCheckpointStatusUpdate(bad, old); len(errs) == 0 {
		t.Errorf("expected errors for invalid condition, got none")
	}
}
