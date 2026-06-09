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
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/apps/replicaset"
)

func mkReplicaSet(tolerations ...api.Toleration) *apps.ReplicaSet {
	return &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"name": "abc"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"name": "abc"}},
				Spec:       podtest.MakePodSpec(podtest.SetTolerations(tolerations...)),
			},
		},
	}
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
				input: mkReplicaSet(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists}),
			},
			"valid toleration key without prefix": {
				input: mkReplicaSet(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists}),
			},
			"invalid toleration key format": {
				input: mkReplicaSet(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists}),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs,
					apitesting.WithSkipGroupVersions("extensions/v1beta1"))
			})
		}
	}
}
