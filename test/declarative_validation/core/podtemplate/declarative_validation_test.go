/*
Copyright 2025 The Kubernetes Authors.

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

package podtemplate

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/core/podtemplate"
	poddeclarativevalidation "k8s.io/kubernetes/test/declarative_validation/core/pod"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func mkPodTemplate(tolerations ...api.Toleration) *api.PodTemplate {
	return &api.PodTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: metav1.NamespaceDefault},
		Template: api.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"a": "b"}},
			Spec:       podtest.MakePodSpec(podtest.SetTolerations(tolerations...)),
		},
	}
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			IsResourceRequest: true,
			Verb:              "create",
		})
		testCases := map[string]struct {
			input        *api.PodTemplate
			expectedErrs field.ErrorList
		}{
			"valid": {
				input: mkPodTemplate(),
			},
			"valid toleration key": {
				input: mkPodTemplate(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists}),
			},
			"valid toleration key without prefix": {
				input: mkPodTemplate(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists}),
			},
			"invalid toleration key format": {
				input: mkPodTemplate(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists}),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs)
			})
		}
		obj := *mkPodTemplate()
		meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
		poddeclarativevalidation.RunDeclarativeValidateEvictionRespondersTestCases(t, ctx, registry.Strategy, field.NewPath("template", "spec"), mkPodTemplate(), func(baseObj *api.PodTemplate, responders []api.EvictionResponder, schedulingGroup *api.PodSchedulingGroup) {
			baseObj.Template.Spec.EvictionResponders = responders
			baseObj.Template.Spec.SchedulingGroup = schedulingGroup
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			Name:              "valid-obj",
			IsResourceRequest: true,
			Verb:              "update",
		})
		updateObj := *mkPodTemplate()
		meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
	}
}

// TestDeclarativeValidateRestoreFrom covers the declarative rules on the pod
// template's spec.restoreFrom (KEP-5823): the referenced PodCheckpoint name is
// required and must be a valid long name. The feature gate is enabled because a
// present restoreFrom is only validated with the gate on (the field is dropped
// in PrepareForCreate otherwise).
func TestDeclarativeValidateRestoreFrom(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			IsResourceRequest: true,
			Verb:              "create",
		})
		testCases := map[string]struct {
			input        *api.PodTemplate
			expectedErrs field.ErrorList
		}{
			"restoreFrom: valid name": {
				input: mkPodTemplateRestoreFrom("valid-checkpoint"),
			},
			"restoreFrom: invalid name format": {
				input: mkPodTemplateRestoreFrom("Invalid-Name"),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("template", "spec", "restoreFrom", "name"), nil, "").WithOrigin("format=k8s-long-name").MarkAlpha(),
				},
			},
			"restoreFrom: empty name": {
				input: mkPodTemplateRestoreFrom(""),
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("template", "spec", "restoreFrom", "name"), "").MarkAlpha(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs)
			})
		}
	}
}

func mkPodTemplateRestoreFrom(name string) *api.PodTemplate {
	pt := mkPodTemplate()
	pt.Template.Spec.RestoreFrom = &api.CheckpointReference{Name: name}
	return pt
}
