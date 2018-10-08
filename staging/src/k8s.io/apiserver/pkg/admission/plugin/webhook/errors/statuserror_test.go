/*
Copyright 2017 The Kubernetes Authors.

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

package errors

import (
	"fmt"
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestToStatusErr(t *testing.T) {
	hookName := "foo"
	deniedBy := fmt.Sprintf("admission webhook %q denied the request", hookName)
	tests := []struct {
		name       string
		attributes mockAttributes
		result     *metav1.Status
		expected   metav1.Status
	}{
		{
			"nil result",
			mockAttributes{"example"},
			nil,
			admission.NewForbidden(mockAttributes{"example"}, fmt.Errorf(deniedBy+" without explanation")).(*apierrors.StatusError).Status(),
		},
		{
			"only message",
			mockAttributes{"example"},
			&metav1.Status{
				Message: "you shall not pass",
			},
			admission.NewForbidden(mockAttributes{"example"}, fmt.Errorf(deniedBy+": you shall not pass")).(*apierrors.StatusError).Status(),
		},
		{
			"only reason",
			mockAttributes{"example"},
			&metav1.Status{
				Reason: metav1.StatusReasonForbidden,
			},
			admission.NewForbidden(mockAttributes{"example"}, fmt.Errorf(deniedBy+": Forbidden")).(*apierrors.StatusError).Status(),
		},
		{
			"message and reason",
			mockAttributes{"example"},
			&metav1.Status{
				Message: "you shall not pass",
				Reason:  metav1.StatusReasonForbidden,
			},
			admission.NewForbidden(mockAttributes{"example"}, fmt.Errorf(deniedBy+": you shall not pass")).(*apierrors.StatusError).Status(),
		},
		{
			"no message, no reason",
			mockAttributes{"example"},
			&metav1.Status{},
			admission.NewForbidden(mockAttributes{"example"}, fmt.Errorf(deniedBy+" without explanation")).(*apierrors.StatusError).Status(),
		},
		{
			"unknown name",
			mockAttributes{},
			&metav1.Status{},
			admission.NewForbidden(mockAttributes{}, fmt.Errorf(deniedBy+" without explanation")).(*apierrors.StatusError).Status(),
		},
		{
			"500 error",
			mockAttributes{},
			&metav1.Status{Code: http.StatusInternalServerError, Message: "internal error"},
			apierrors.NewInternalError(fmt.Errorf(deniedBy + ": internal error")).Status(),
		},
	}
	for _, test := range tests {
		err := ToStatusErr(test.attributes, hookName, test.result)
		if err == nil || !equality.Semantic.DeepEqual(err.Status(), test.expected) {
			t.Errorf("%s: unexpected error (got A, expected B):\n%s", test.name, diff.ObjectDiff(err.Status(), test.expected))
		}
	}
}

type mockAttributes struct {
	name string
}

var _ admission.Attributes = &mockAttributes{}

func (a mockAttributes) GetKind() schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: "foo", Version: "v1", Kind: "Bar"}
}

func (a mockAttributes) GetNamespace() string {
	return "default"
}

func (a mockAttributes) GetName() string {
	return a.name
}

func (a mockAttributes) GetResource() schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: "foo", Version: "v1", Resource: "bars"}
}

func (a mockAttributes) GetSubresource() string {
	return ""
}

func (a mockAttributes) GetOperation() admission.Operation {
	return admission.Create
}

func (a mockAttributes) IsDryRun() bool {
	return false
}

func (a mockAttributes) GetObject() runtime.Object {
	return nil
}

func (a mockAttributes) GetOldObject() runtime.Object {
	return nil
}

func (a mockAttributes) GetUserInfo() user.Info {
	return nil
}

func (a mockAttributes) AddAnnotation(key, value string) error {
	return nil
}
