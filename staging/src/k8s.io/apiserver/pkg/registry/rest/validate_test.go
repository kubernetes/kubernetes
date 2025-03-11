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

package rest

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestValidateDeclaratively(t *testing.T) {
	valid := &Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
	}

	invalidRestartPolicy := &Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
		RestartPolicy: "INVALID",
	}

	invalidRestartPolicyErr := field.Invalid(field.NewPath("spec", "restartPolicy"), "", "Invalid value").WithOrigin("invalid-test")
	mutatedRestartPolicyErr := field.Invalid(field.NewPath("spec", "restartPolicy"), "", "Immutable field").WithOrigin("immutable-test")
	invalidStatusErr := field.Invalid(field.NewPath("status", "conditions"), "", "Invalid condition").WithOrigin("invalid-condition")
	invalidIfOptionErr := field.Invalid(field.NewPath("spec", "restartPolicy"), "", "Invalid when option is set").WithOrigin("invalid-when-option-set")
	invalidSubresourceErr := field.InternalError(nil, fmt.Errorf("unexpected error parsing subresource path: %w", fmt.Errorf("invalid subresource path: %s", "invalid/status")))

	testCases := []struct {
		name        string
		object      runtime.Object
		oldObject   runtime.Object
		subresource string
		options     sets.Set[string]
		expected    field.ErrorList
	}{
		{
			name:     "create",
			object:   invalidRestartPolicy,
			expected: field.ErrorList{invalidRestartPolicyErr},
		},
		{
			name:      "update",
			object:    invalidRestartPolicy,
			oldObject: valid,
			expected:  field.ErrorList{invalidRestartPolicyErr, mutatedRestartPolicyErr},
		},
		{
			name:        "update subresource",
			subresource: "/status",
			object:      valid,
			oldObject:   valid,
			expected:    field.ErrorList{invalidStatusErr},
		},
		{
			name:        "invalid subresource",
			subresource: "invalid/status",
			object:      valid,
			oldObject:   valid,
			expected:    field.ErrorList{invalidSubresourceErr},
		},
		{
			name:     "update with option",
			options:  sets.New("option1"),
			object:   valid,
			expected: field.ErrorList{invalidIfOptionErr},
		},
	}

	ctx := context.Background()

	internalGV := schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal}
	v1GV := schema.GroupVersion{Group: "", Version: "v1"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypes(internalGV, &Pod{})
	scheme.AddKnownTypes(v1GV, &v1.Pod{})

	scheme.AddValidationFunc(&v1.Pod{}, func(ctx context.Context, op operation.Operation, object, oldObject interface{}, subresources ...string) field.ErrorList {
		results := field.ErrorList{}
		if op.Options.Has("option1") {
			results = append(results, invalidIfOptionErr)
		}
		if len(subresources) == 1 && subresources[0] == "status" {
			results = append(results, invalidStatusErr)
		}
		if op.Type == operation.Update && object.(*v1.Pod).Spec.RestartPolicy != oldObject.(*v1.Pod).Spec.RestartPolicy {
			results = append(results, mutatedRestartPolicyErr)
		}
		if object.(*v1.Pod).Spec.RestartPolicy == "INVALID" {
			results = append(results, invalidRestartPolicyErr)
		}
		return results
	})
	err := scheme.AddConversionFunc(&Pod{}, &v1.Pod{}, func(a, b interface{}, scope conversion.Scope) error {
		if in, ok := a.(*Pod); ok {
			if out, ok := b.(*v1.Pod); ok {
				out.APIVersion = in.APIVersion
				out.Kind = in.Kind
				out.Spec.RestartPolicy = v1.RestartPolicy(in.RestartPolicy)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range testCases {
		ctx = genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
			APIGroup:    "",
			APIVersion:  "v1",
			Subresource: tc.subresource,
		})
		t.Run(tc.name, func(t *testing.T) {
			var results field.ErrorList
			if tc.oldObject == nil {
				results = ValidateDeclaratively(ctx, tc.options, scheme, tc.object)
			} else {
				results = ValidateUpdateDeclaratively(ctx, tc.options, scheme, tc.object, tc.oldObject)
			}
			matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			matcher.Test(t, tc.expected, results)
		})
	}
}

// Fake internal pod type, since core.Pod cannot be imported by this package
type Pod struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	RestartPolicy     string `json:"restartPolicy"`
}

func (Pod) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }

func (p Pod) DeepCopyObject() runtime.Object {
	return &Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: p.APIVersion,
			Kind:       p.Kind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      p.Name,
			Namespace: p.Namespace,
		},
		RestartPolicy: p.RestartPolicy,
	}
}
