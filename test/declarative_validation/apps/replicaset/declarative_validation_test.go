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
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/apps/replicaset"
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

func tweakSelector(selector *metav1.LabelSelector) func(*apps.ReplicaSet) {
	return func(rs *apps.ReplicaSet) {
		rs.Spec.Selector = selector
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
			APIGroup:          "apps",
			APIVersion:        apiVersion,
			IsResourceRequest: true,
			Verb:              "create",
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
			"missing selector": {
				input: mkReplicaSet(tweakSelector(nil)),
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("spec", "selector"), "").MarkAlpha(),
					field.Invalid(field.NewPath("spec", "template", "metadata", "labels"), nil, "").MarkFromImperative(),
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
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "apps",
		APIVersion:        apiVersion,
		IsResourceRequest: true,
		Verb:              "update",
	})
	meta.RunObjectMetaUpdateTestCases(t, ctx, mkReplicaSet(), registry.Strategy,
		meta.WithStringentFinalizerValidation(),
	)

	testCases := map[string]struct {
		old          *apps.ReplicaSet
		update       *apps.ReplicaSet
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    mkReplicaSet(),
			update: mkReplicaSet(),
		},
		"selector changed": {
			old:    mkReplicaSet(),
			update: mkReplicaSet(tweakSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"name": "different"}})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "template", "metadata", "labels"), nil, "").MarkFromImperative(),
				field.Invalid(field.NewPath("spec", "selector"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, tc.update, tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
}
