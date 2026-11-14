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

package podcheckpoint

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/node"
	registry "k8s.io/kubernetes/pkg/registry/node/podcheckpoint"
)

func TestDeclarativeValidateRestoreReservations(t *testing.T) {
	consumer := node.PodCheckpointConsumerReference{Name: "restore", UID: "restore-uid"}
	path := field.NewPath("status", "activeRestores")
	generationPath := field.NewPath("status", "restoreGeneration")
	tests := map[string]struct {
		mutate       func(*node.PodCheckpoint, *node.PodCheckpoint)
		expectedErrs field.ErrorList
	}{
		"ready consumer": {},
		"optional fields can be unset": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.RestoreGeneration, newPC.Status.RestoreGeneration = nil, nil
				newPC.Status.ActiveRestores = nil
			},
		},
		"missing name": {
			mutate:       func(_, newPC *node.PodCheckpoint) { newPC.Status.ActiveRestores[0].Name = "" },
			expectedErrs: field.ErrorList{field.Required(path.Index(0).Child("name"), "")},
		},
		"invalid name": {
			mutate:       func(_, newPC *node.PodCheckpoint) { newPC.Status.ActiveRestores[0].Name = "Invalid_Name" },
			expectedErrs: field.ErrorList{field.Invalid(path.Index(0).Child("name"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"name too long": {
			mutate:       func(_, newPC *node.PodCheckpoint) { newPC.Status.ActiveRestores[0].Name = strings.Repeat("a", 254) },
			expectedErrs: field.ErrorList{field.Invalid(path.Index(0).Child("name"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"empty UID": {
			mutate:       func(_, newPC *node.PodCheckpoint) { newPC.Status.ActiveRestores[0].UID = "" },
			expectedErrs: field.ErrorList{field.Required(path.Index(0).Child("uid"), "")},
		},
		"duplicate UID": {
			mutate: func(_, newPC *node.PodCheckpoint) {
				newPC.Status.ActiveRestores = append(newPC.Status.ActiveRestores, node.PodCheckpointConsumerReference{Name: "another-name", UID: consumer.UID})
			},
			expectedErrs: field.ErrorList{field.Duplicate(path.Index(1), nil)},
		},
		"same name with different UIDs": {
			mutate: func(_, newPC *node.PodCheckpoint) {
				newPC.Status.ActiveRestores = append(newPC.Status.ActiveRestores, node.PodCheckpointConsumerReference{Name: consumer.Name, UID: "another-uid"})
			},
		},
		"consumers can be reordered": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.ActiveRestores = []node.PodCheckpointConsumerReference{consumer, {Name: "other", UID: "other-uid"}}
				newPC.Status.ActiveRestores = []node.PodCheckpointConsumerReference{oldPC.Status.ActiveRestores[1], consumer}
			},
		},
		"new invalid name after reorder": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.ActiveRestores = []node.PodCheckpointConsumerReference{consumer, {Name: "other", UID: "other-uid"}}
				newPC.Status.ActiveRestores = []node.PodCheckpointConsumerReference{oldPC.Status.ActiveRestores[1], {Name: "Invalid_Name", UID: consumer.UID}}
			},
			expectedErrs: field.ErrorList{field.Invalid(path.Index(1).Child("name"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"unchanged invalid name is ratcheted after reorder": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.ActiveRestores = []node.PodCheckpointConsumerReference{{Name: "Invalid_Name", UID: consumer.UID}, {Name: "other", UID: "other-uid"}}
				newPC.Status.ActiveRestores = []node.PodCheckpointConsumerReference{oldPC.Status.ActiveRestores[1], oldPC.Status.ActiveRestores[0]}
			},
		},
		"missing protection": {
			mutate:       func(oldPC, newPC *node.PodCheckpoint) { oldPC.Finalizers, newPC.Finalizers = nil, nil },
			expectedErrs: field.ErrorList{field.Forbidden(path.Index(0), "").MarkFromImperative()},
		},
		"not ready": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.Conditions[0].Status = metav1.ConditionFalse
				oldPC.Status.Conditions[0].Reason = node.PodCheckpointReasonInProgress
				newPC.Status.Conditions = slices.Clone(oldPC.Status.Conditions)
			},
			expectedErrs: field.ErrorList{field.Forbidden(path.Index(0), "").MarkFromImperative()},
		},
		"unassigned checkpoint": {
			mutate:       func(oldPC, newPC *node.PodCheckpoint) { oldPC.Status.NodeName, newPC.Status.NodeName = nil, nil },
			expectedErrs: field.ErrorList{field.Forbidden(path.Index(0), "").MarkFromImperative()},
		},
		"new consumer during deletion": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.DeletionTimestamp = new(metav1.Now())
				newPC.DeletionTimestamp = oldPC.DeletionTimestamp
			},
			expectedErrs: field.ErrorList{field.Forbidden(path.Index(0), "").MarkFromImperative()},
		},
		"existing consumer during deletion": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.ActiveRestores = slices.Clone(newPC.Status.ActiveRestores)
				oldPC.DeletionTimestamp = new(metav1.Now())
				newPC.DeletionTimestamp = oldPC.DeletionTimestamp
			},
		},
		"release during deletion": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.ActiveRestores = slices.Clone(newPC.Status.ActiveRestores)
				oldPC.DeletionTimestamp = new(metav1.Now())
				newPC.DeletionTimestamp = oldPC.DeletionTimestamp
				newPC.Status.ActiveRestores = nil
				newPC.Status.RestoreGeneration = new(*oldPC.Status.RestoreGeneration + 1)
			},
		},
		"generation can increase": {
			mutate: func(_, newPC *node.PodCheckpoint) { newPC.Status.RestoreGeneration = new(int64(2)) },
		},
		"generation can be initialized to zero": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.RestoreGeneration = nil
				newPC.Status.RestoreGeneration = new(int64(0))
			},
		},
		"generation cannot be cleared": {
			mutate:       func(_, newPC *node.PodCheckpoint) { newPC.Status.RestoreGeneration = nil },
			expectedErrs: field.ErrorList{field.Invalid(generationPath, nil, "").WithOrigin("update")},
		},
		"zero generation cannot be cleared": {
			mutate: func(oldPC, newPC *node.PodCheckpoint) {
				oldPC.Status.RestoreGeneration = new(int64(0))
				newPC.Status.RestoreGeneration = nil
			},
			expectedErrs: field.ErrorList{field.Invalid(generationPath, nil, "").WithOrigin("update")},
		},
		"generation cannot decrease to zero": {
			mutate:       func(_, newPC *node.PodCheckpoint) { newPC.Status.RestoreGeneration = new(int64(0)) },
			expectedErrs: field.ErrorList{field.Invalid(generationPath, nil, "").WithOrigin("monotonic")},
		},
		"generation cannot decrease to a nonzero value": {
			mutate:       func(oldPC, _ *node.PodCheckpoint) { oldPC.Status.RestoreGeneration = new(int64(2)) },
			expectedErrs: field.ErrorList{field.Invalid(generationPath, nil, "").WithOrigin("monotonic")},
		},
		"negative generation": {
			mutate: func(_, newPC *node.PodCheckpoint) { newPC.Status.RestoreGeneration = new(int64(-1)) },
			expectedErrs: field.ErrorList{
				field.Invalid(generationPath, nil, "").WithOrigin("minimum"),
				field.Invalid(generationPath, nil, "").WithOrigin("monotonic"),
			},
		},
		"at capacity": {
			mutate: func(_, newPC *node.PodCheckpoint) {
				newPC.Status.ActiveRestores = nil
				for i := range node.PodCheckpointActiveRestoresMaxSize {
					newPC.Status.ActiveRestores = append(newPC.Status.ActiveRestores, node.PodCheckpointConsumerReference{Name: fmt.Sprintf("restore-%d", i), UID: types.UID(fmt.Sprintf("uid-%d", i))})
				}
			},
		},
		"over capacity": {
			mutate: func(_, newPC *node.PodCheckpoint) {
				newPC.Status.ActiveRestores = nil
				for i := range node.PodCheckpointActiveRestoresMaxSize + 1 {
					newPC.Status.ActiveRestores = append(newPC.Status.ActiveRestores, node.PodCheckpointConsumerReference{Name: fmt.Sprintf("restore-%d", i), UID: types.UID(fmt.Sprintf("uid-%d", i))})
				}
			},
			expectedErrs: field.ErrorList{field.TooMany(path, node.PodCheckpointActiveRestoresMaxSize+1, node.PodCheckpointActiveRestoresMaxSize).WithOrigin("maxItems")},
		},
	}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := requestContext(apiVersion, "update", "status")
			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					oldPC := mkPodCheckpoint()
					oldPC.ResourceVersion = "1"
					oldPC.Finalizers = []string{node.PodCheckpointRestoreProtectionFinalizer}
					oldPC.Status = node.PodCheckpointStatus{
						NodeName:          new("node-a"),
						RestoreGeneration: new(int64(1)),
						Conditions: []metav1.Condition{{
							Type:               node.PodCheckpointConditionReady,
							Status:             metav1.ConditionTrue,
							Reason:             node.PodCheckpointReasonCompleted,
							LastTransitionTime: metav1.Now(),
						}},
					}
					newPC := oldPC.DeepCopy()
					newPC.Status.ActiveRestores = []node.PodCheckpointConsumerReference{consumer}
					if tc.mutate != nil {
						tc.mutate(oldPC, newPC)
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPC, oldPC, registry.StatusStrategy, tc.expectedErrs, apitesting.WithSubResources("status"))
				})
			}
		})
	}
}
