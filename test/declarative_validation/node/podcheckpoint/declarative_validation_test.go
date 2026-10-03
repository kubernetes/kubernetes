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
	"context"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
	registry "k8s.io/kubernetes/pkg/registry/node/podcheckpoint"
	poddeclarativevalidation "k8s.io/kubernetes/test/declarative_validation/core/pod"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
)

func mkPodCheckpoint() *node.PodCheckpoint {
	return &node.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-1", Namespace: "default"},
		Spec: node.PodCheckpointSpec{
			SourcePod: &node.PodReference{Name: "my-pod"},
		},
	}
}

func requestContext(apiVersion, verb, subresource string) context.Context {
	return genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "node.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "podcheckpoints",
		Subresource:       subresource,
		IsResourceRequest: true,
		Verb:              verb,
	})
}

func TestDeclarativeValidate(t *testing.T) {
	specPath := field.NewPath("spec")
	testCases := map[string]struct {
		mutate       func(*node.PodCheckpoint)
		expectedErrs field.ErrorList
	}{
		"valid with unset timeout and UID": {},
		"missing checkpoint name": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Name = "" },
			expectedErrs: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "").MarkFromImperative()},
		},
		"valid minimum timeout": {
			mutate: func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](1) },
		},
		"valid maximum timeout": {
			mutate: func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](3600) },
		},
		"valid source pod DNS subdomain": {
			mutate: func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.Name = "my-app.example" },
		},
		"valid opaque source pod UID": {
			mutate: func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = ptr.To(types.UID("opaque-uid")) },
		},
		"valid checkpoint options": {
			mutate: func(pc *node.PodCheckpoint) {
				pc.Spec.CheckpointOptions = map[string]string{"example.runtime/mode": "incremental"}
			},
		},
		"missing source pod": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod = nil },
			expectedErrs: field.ErrorList{field.Required(specPath.Child("sourcePod"), "")},
		},
		"missing source pod name": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.Name = "" },
			expectedErrs: field.ErrorList{field.Required(specPath.Child("sourcePod", "name"), "")},
		},
		"invalid source pod name": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.Name = "Not_A_Valid_Name" },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("sourcePod", "name"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"source pod name too long": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.Name = strings.Repeat("a", 254) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("sourcePod", "name"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"empty source pod UID": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = ptr.To(types.UID("")) },
			expectedErrs: field.ErrorList{field.TooShort(specPath.Child("sourcePod", "uid"), "", 1).WithOrigin("minLength")},
		},
		"negative timeout": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](-1) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("minimum")},
		},
		"zero timeout is not unset": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](0) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("minimum")},
		},
		"timeout above maximum": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](3601) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("maximum")},
		},
	}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := requestContext(apiVersion, "create", "")
			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					pc := mkPodCheckpoint()
					if tc.mutate != nil {
						tc.mutate(pc)
					}
					apitesting.VerifyValidationEquivalence(t, ctx, pc, registry.Strategy, tc.expectedErrs)
				})
			}
			poddeclarativevalidation.RunDeclarativeValidateRuntimeOptionsTestCases(t, ctx, registry.Strategy, specPath.Child("checkpointOptions"), mkPodCheckpoint(), func(pc *node.PodCheckpoint, options map[string]string) {
				pc.Spec.CheckpointOptions = options
			})
			meta.RunObjectMetaTestCases(t, ctx, mkPodCheckpoint(), registry.Strategy)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	specPath := field.NewPath("spec")
	testCases := map[string]struct {
		mutateOld    func(*node.PodCheckpoint)
		mutate       func(*node.PodCheckpoint)
		expectedErrs field.ErrorList
	}{
		"unchanged": {},
		"unchanged pinned source pod": {
			mutateOld: func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = ptr.To(types.UID("opaque-uid")) },
		},
		"unchanged checkpoint options": {
			mutateOld: func(pc *node.PodCheckpoint) {
				pc.Spec.CheckpointOptions = map[string]string{"example.runtime/mode": "incremental"}
			},
		},
		"unchanged timeout": {
			mutateOld: func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](30) },
			mutate:    func(pc *node.PodCheckpoint) { pc.Labels = map[string]string{"updated": "true"} },
		},
		"timeout cannot be added": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](30) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("immutable")},
		},
		"timeout cannot be changed": {
			mutateOld:    func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](30) },
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](60) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("immutable")},
		},
		"timeout cannot be removed": {
			mutateOld:    func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](30) },
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = nil },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("immutable")},
		},
		"unchanged legacy timeout is ratcheted": {
			mutateOld: func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](0) },
			mutate:    func(pc *node.PodCheckpoint) { pc.Labels = map[string]string{"updated": "true"} },
		},
		"source pod name is immutable": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.Name = "other-pod" },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("sourcePod"), nil, "").WithOrigin("immutable")},
		},
		"source pod UID cannot be added": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = ptr.To(types.UID("opaque-uid")) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("sourcePod"), nil, "").WithOrigin("immutable")},
		},
		"source pod UID cannot be changed": {
			mutateOld:    func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = ptr.To(types.UID("old-uid")) },
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = ptr.To(types.UID("new-uid")) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("sourcePod"), nil, "").WithOrigin("immutable")},
		},
		"source pod UID cannot be removed": {
			mutateOld:    func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = ptr.To(types.UID("opaque-uid")) },
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.SourcePod.UID = nil },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("sourcePod"), nil, "").WithOrigin("immutable")},
		},
		"source pod cannot be removed": {
			mutate: func(pc *node.PodCheckpoint) { pc.Spec.SourcePod = nil },
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("sourcePod"), nil, "").WithOrigin("immutable"),
				field.Required(specPath.Child("sourcePod"), ""),
			},
		},
		"checkpoint options are immutable": {
			mutate: func(pc *node.PodCheckpoint) {
				pc.Spec.CheckpointOptions = map[string]string{"example.runtime/mode": "incremental"}
			},
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("checkpointOptions"), nil, "").WithOrigin("immutable")},
		},
		"checkpoint options cannot be removed": {
			mutateOld: func(pc *node.PodCheckpoint) {
				pc.Spec.CheckpointOptions = map[string]string{"example.runtime/mode": "incremental"}
			},
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.CheckpointOptions = nil },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("checkpointOptions"), nil, "").WithOrigin("immutable")},
		},
		"zero timeout is not unset": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](0) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("immutable")},
		},
		"timeout above maximum": {
			mutate:       func(pc *node.PodCheckpoint) { pc.Spec.TimeoutSeconds = ptr.To[int32](3601) },
			expectedErrs: field.ErrorList{field.Invalid(specPath.Child("timeoutSeconds"), nil, "").WithOrigin("immutable")},
		},
	}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := requestContext(apiVersion, "update", "")
			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					oldPC := mkPodCheckpoint()
					oldPC.ResourceVersion = "1"
					if tc.mutateOld != nil {
						tc.mutateOld(oldPC)
					}
					newPC := oldPC.DeepCopy()
					if tc.mutate != nil {
						tc.mutate(newPC)
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPC, oldPC, registry.Strategy, tc.expectedErrs)
				})
			}
			meta.RunObjectMetaUpdateTestCases(t, ctx, mkPodCheckpoint(), registry.Strategy)
		})
	}
}

func TestValidationWithDeclarativeValidationBetaDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiserverfeatures.DeclarativeValidationBeta, false)
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			pc := mkPodCheckpoint()
			pc.Spec.SourcePod = nil
			pc.Spec.TimeoutSeconds = ptr.To[int32](0)
			errMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			errMatcher.Test(t, field.ErrorList{
				field.Required(field.NewPath("spec", "sourcePod"), ""),
				field.Invalid(field.NewPath("spec", "timeoutSeconds"), nil, "").WithOrigin("minimum"),
			}, rest.ValidateCreate(requestContext(apiVersion, "create", ""), pc, registry.Strategy))

			oldPC := mkPodCheckpoint()
			oldPC.ResourceVersion = "1"
			newPC := oldPC.DeepCopy()
			newPC.Spec.TimeoutSeconds = ptr.To[int32](30)
			newPC.Spec.SourcePod.Name = "other-pod"
			newPC.Spec.CheckpointOptions = map[string]string{"example.runtime/mode": "incremental"}
			errMatcher.Test(t, field.ErrorList{
				field.Invalid(field.NewPath("spec", "timeoutSeconds"), nil, "").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "sourcePod"), nil, "").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "checkpointOptions"), nil, "").WithOrigin("immutable"),
			}, rest.ValidateUpdate(requestContext(apiVersion, "update", ""), newPC, oldPC, registry.Strategy))
		})
	}
}

func TestStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := requestContext(apiVersion, "update", "status")
			oldPC := mkPodCheckpoint()
			oldPC.ResourceVersion = "1"
			newPC := oldPC.DeepCopy()
			newPC.Status.Conditions = []metav1.Condition{{
				Type:               node.PodCheckpointConditionReady,
				Status:             metav1.ConditionTrue,
				Reason:             node.PodCheckpointReasonCompleted,
				LastTransitionTime: metav1.Now(),
			}}
			// Captured templates are sanitized snapshots, not new Pods to admit.
			newPC.Status.CheckpointedPodTemplate = &core.PodTemplateSpec{}
			newPC.Spec.SourcePod = nil
			registry.StatusStrategy.PrepareForUpdate(ctx, newPC, oldPC)
			if errs := rest.ValidateUpdate(ctx, newPC, oldPC, registry.StatusStrategy); len(errs) != 0 {
				t.Fatalf("valid status update failed: %v", errs)
			}

			newPC.Status.Conditions[0].Status = "Maybe"
			errs := rest.ValidateUpdate(ctx, newPC, oldPC, registry.StatusStrategy)
			field.ErrorMatcher{}.ByType().ByField().Test(t, field.ErrorList{
				field.NotSupported(field.NewPath("status", "conditions").Index(0).Child("status"), "Maybe", []string{"False", "True", "Unknown"}),
			}, errs)
		})
	}
}
