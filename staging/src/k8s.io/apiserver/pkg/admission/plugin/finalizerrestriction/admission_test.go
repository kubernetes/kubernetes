/*
Copyright 2019 The Kubernetes Authors.

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

package finalizerrestriction

import (
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

type simpleAuthorizer struct {
	allowedFinalizers []string
}

func (s *simpleAuthorizer) Authorize(a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	if a.GetVerb() != "finalize" {
		return authorizer.DecisionDeny, "", fmt.Errorf("unsupported verb")
	}
	for a.GetUser() == nil {
		return authorizer.DecisionDeny, "", fmt.Errorf("user info is missing")
	}
	for _, finalizer := range s.allowedFinalizers {
		if finalizer == a.GetName() {
			return authorizer.DecisionAllow, "", nil
		}
	}
	return authorizer.DecisionDeny, "not allowed", nil
}

func TestAdmit(t *testing.T) {
	var tests = []struct {
		name        string
		operation   admission.Operation
		object      runtime.Object
		oldObject   runtime.Object
		subresource string
		allowed     bool
	}{
		{
			name:      "ignore non create and non update operation",
			operation: admission.Delete,
			allowed:   true,
		},
		{
			name:        "ignore subresource request",
			operation:   admission.Update,
			subresource: "status",
			allowed:     true,
		},
		{
			name:      "allow finalizing foo in creation",
			operation: admission.Create,
			object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo"},
				},
			},
			oldObject: nil,
			allowed:   true,
		},
		{
			name:      "disallow finalizing bar in creation",
			operation: admission.Create,
			object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo", "bar"},
				},
			},
			oldObject: nil,
			allowed:   false,
		},
		{
			name:      "allow finalizing when adding foo",
			operation: admission.Create,
			object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{},
				},
			},
			oldObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo"},
				},
			},
			allowed: true,
		},
		{
			name:      "allow finalizing when removing foo",
			operation: admission.Create,
			object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo"},
				},
			},
			oldObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{},
				},
			},
			allowed: true,
		},
		{
			name:      "disallow finalizing when adding bar",
			operation: admission.Create,
			object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo"},
				},
			},
			oldObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo", "bar"},
				},
			},
			allowed: false,
		},
		{
			name:      "disallow finalizing when removing bar",
			operation: admission.Create,
			object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo", "bar"},
				},
			},
			oldObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Finalizers: []string{"foo"},
				},
			},
			allowed: false,
		},
	}

	admitter := &FinalizerRestriction{
		authorizer: &simpleAuthorizer{
			allowedFinalizers: []string{"foo"},
		},
	}
	for _, test := range tests {
		err := admitter.Admit(admission.NewAttributesRecord(
			test.object,
			test.oldObject,
			schema.GroupVersionKind{},
			"",
			"",
			schema.GroupVersionResource{},
			test.subresource,
			test.operation,
			false,
			&user.DefaultInfo{Name: "bar"},
		), nil)
		if test.allowed && err != nil {
			t.Errorf("%s: should be alloed but got error: %s", test.name, err)
		}
		if !test.allowed && err == nil {
			t.Errorf("%s: should be disallowed", test.name)
		}
	}
}
