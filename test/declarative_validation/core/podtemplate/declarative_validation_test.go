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
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/podtemplate"
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
			APIGroup:   "",
			APIVersion: apiVersion,
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
	}
}
