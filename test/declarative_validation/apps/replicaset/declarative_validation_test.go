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

package replicaset

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/apps/replicaset"
	poddeclarativevalidation "k8s.io/kubernetes/test/declarative_validation/core/pod"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func tweakPodSpec(podTweaks ...podtest.Tweak) func(*apps.ReplicaSet) {
	return func(rs *apps.ReplicaSet) {
		pod := &api.Pod{
			Spec: rs.Spec.Template.Spec,
		}
		for _, tweak := range podTweaks {
			tweak(pod)
		}
		rs.Spec.Template.Spec = pod.Spec
	}
}

func mkReplicaSet(tweaks ...func(*apps.ReplicaSet)) *apps.ReplicaSet {
	rs := &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"name": "abc"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"name": "abc"}},
				Spec:       podtest.MakePodSpec(),
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(rs)
	}
	return rs
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: apiVersion,
		})
		testCases := map[string]struct {
			input        *apps.ReplicaSet
			expectedErrs field.ErrorList
		}{
			"valid": {
				input: mkReplicaSet(),
			},
			"valid toleration key": {
				input: mkReplicaSet(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists}))),
			},
			"valid toleration key without prefix": {
				input: mkReplicaSet(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists}))),
			},
			"invalid toleration key format": {
				input: mkReplicaSet(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists}))),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs)
			})
		}
		meta.RunObjectMetaTestCases(t, ctx, mkReplicaSet(), registry.Strategy,
			meta.WithStringentFinalizerValidation(),
		)
		poddeclarativevalidation.RunDeclarativeValidateEvictionRespondersTestCases(t, ctx, registry.Strategy, field.NewPath("spec", "template", "spec"), mkReplicaSet(), func(baseObj *apps.ReplicaSet, responders []api.EvictionResponder, schedulingGroup *api.PodSchedulingGroup) {
			baseObj.Spec.Template.Spec.EvictionResponders = responders
			baseObj.Spec.Template.Spec.SchedulingGroup = schedulingGroup
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: apiVersion,
			Name:       "abc",
			Verb:       "update",
		})
		meta.RunObjectMetaUpdateTestCases(t, ctx, mkReplicaSet(), registry.Strategy,
			meta.WithStringentFinalizerValidation(),
		)
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
			APIGroup:   "apps",
			APIVersion: apiVersion,
		})
		testCases := map[string]struct {
			input        *apps.ReplicaSet
			expectedErrs field.ErrorList
		}{
			"restoreFrom: valid name": {
				input: mkReplicaSet(tweakPodSpec(podtest.SetRestoreFrom("valid-checkpoint"))),
			},
			"restoreFrom: invalid name format": {
				input: mkReplicaSet(tweakPodSpec(podtest.SetRestoreFrom("Invalid-Name"))),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "template", "spec", "restoreFrom", "name"), nil, "").WithOrigin("format=k8s-long-name").MarkAlpha(),
				},
			},
			"restoreFrom: empty name": {
				input: mkReplicaSet(tweakPodSpec(podtest.SetRestoreFrom(""))),
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("spec", "template", "spec", "restoreFrom", "name"), "").MarkAlpha(),
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
