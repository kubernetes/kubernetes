/*
Copyright 2024 The Kubernetes Authors.

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

package mutating

import (
	"context"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	plugincel "k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/utils/pointer"
)

// TestCompilation is an open-box test of mutatingEvaluator.compile
// However, the result is a set of CEL programs, manually invoke them to assert
// on the results.
func TestCompilation(t *testing.T) {
	for _, tc := range []struct {
		name           string
		policy         *Policy
		object         runtime.Object
		oldObject      runtime.Object
		expectedErr    string
		expectedResult map[string]any
	}{
		{
			name: "refer to object",
			policy: &Policy{
				Spec: MutatingAdmissionPolicySpec{Mutations: []Mutation{
					{
						PatchType: "Apply",
						Expression: `Object{
							spec: Object.spec{
								replicas: object.spec.replicas % 2 == 0?object.spec.replicas + 1:object.spec.replicas
							}
						}`,
					},
				}},
			},
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: pointer.Int32(2)}},
			expectedResult: map[string]any{
				"spec": map[string]any{
					"replicas": int64(3),
				},
			},
		},
		{
			name: "refer to oldObject",
			policy: &Policy{
				Spec: MutatingAdmissionPolicySpec{Mutations: []Mutation{
					{
						PatchType: "Apply",
						Expression: `Object{
							spec: Object.spec{
								replicas: oldObject.spec.replicas % 2 == 0?oldObject.spec.replicas + 1:oldObject.spec.replicas
							}
						}`,
					},
				}},
			},
			object:    &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: pointer.Int32(1)}},
			oldObject: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: pointer.Int32(2)}},
			expectedResult: map[string]any{
				"spec": map[string]any{
					"replicas": int64(3),
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := &evaluator{policy: tc.policy}
			c, err := e.compile(plugincel.OptionalVariableDeclarations{HasParams: e.hasParams()})
			if err != nil {
				if tc.expectedErr == "" {
					t.Fatalf("unexpected error: %v", err)
				}
			}
			a := &activation{}
			_ = a.SetObject(tc.object)
			_ = a.SetOldObject(tc.oldObject)
			p := c.programs[0]
			v, _, err := p.ContextEval(context.Background(), a)
			if !reflect.DeepEqual(tc.expectedResult, v.Value()) {
				t.Errorf("unexpected result, expected %v but got %v", tc.expectedResult, v.Value())
			}
		})
	}
}
