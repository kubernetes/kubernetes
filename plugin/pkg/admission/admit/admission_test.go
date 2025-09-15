/*
Copyright 2014 The Kubernetes Authors.

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

package admit

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestAdmissionNonNilAttribute(t *testing.T) {
	handler := admissiontesting.WithReinvocationTesting(t, NewAlwaysAdmit().(*alwaysAdmit))
	err := handler.Admit(context.TODO(), admission.NewAttributesRecord(nil, nil, api.Kind("kind").WithVersion("version"), "namespace", "name", api.Resource("resource").WithVersion("version"), "subresource", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
}

func TestAdmissionNilAttribute(t *testing.T) {
	handler := NewAlwaysAdmit()
	err := handler.(*alwaysAdmit).Admit(context.TODO(), nil, nil)
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
}

func TestHandles(t *testing.T) {
	handler := NewAlwaysAdmit()
	tests := []admission.Operation{admission.Create, admission.Connect, admission.Update, admission.Delete}

	for _, test := range tests {
		if !handler.Handles(test) {
			t.Errorf("Expected handling all operations, including: %v", test)
		}
	}
}
