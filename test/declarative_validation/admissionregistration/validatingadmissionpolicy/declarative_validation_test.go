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

package validatingadmissionpolicy

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	registry "k8s.io/kubernetes/pkg/registry/admissionregistration/validatingadmissionpolicy"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:    "admissionregistration.k8s.io",
				APIVersion:  apiVersion,
				Resource:    "validatingadmissionpolicies",
				Subresource: "status",
			})

			strategy := registry.NewStatusStrategy(registry.NewStrategy(nil, nil))
			meta.RunConditionTestCases(t, ctx, field.NewPath("status", "conditions"), &admissionregistration.ValidatingAdmissionPolicy{}, strategy, func(obj *admissionregistration.ValidatingAdmissionPolicy, c []metav1.Condition) {
				*obj = admissionregistration.ValidatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "valid-policy", ResourceVersion: "1"},
					Spec:       admissionregistration.ValidatingAdmissionPolicySpec{},
					Status: admissionregistration.ValidatingAdmissionPolicyStatus{
						Conditions: c,
					},
				}
			})
		})
	}
}
