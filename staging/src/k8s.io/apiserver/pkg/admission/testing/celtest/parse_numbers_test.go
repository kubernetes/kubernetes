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

package celtest

import (
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
)

func TestParseAdmissionInputUsesRegisteredTypes(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	input, err := ParseAdmissionInput(`
object:
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: d
  spec:
    replicas: 4
oldObject:
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: d
  spec:
    replicas: 2
params:
  threshold: 2
`)
	if err != nil {
		t.Fatalf("ParseAdmissionInput() error: %v", err)
	}
	deployment, ok := input.object.(*appsv1.Deployment)
	if !ok {
		t.Fatalf("object type = %T, want *appsv1.Deployment", input.object)
	}
	if deployment.Spec.Replicas == nil || *deployment.Spec.Replicas != 4 {
		t.Fatalf("object.spec.replicas = %v, want 4", deployment.Spec.Replicas)
	}
	oldDeployment, ok := input.oldObject.(*appsv1.Deployment)
	if !ok {
		t.Fatalf("oldObject type = %T, want *appsv1.Deployment", input.oldObject)
	}
	if oldDeployment.Spec.Replicas == nil || *oldDeployment.Spec.Replicas != 2 {
		t.Fatalf("oldObject.spec.replicas = %v, want 2", oldDeployment.Spec.Replicas)
	}

	tests := []struct {
		name       string
		expression string
	}{
		{name: "modulo", expression: "object.spec.replicas % 2 == 0"},
		{name: "comparison", expression: "object.spec.replicas > 3"},
		{name: "old object", expression: "oldObject.spec.replicas == 2"},
		{name: "modulo against unstructured params", expression: "object.spec.replicas % params.threshold == 0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policy := &AdmissionPolicy{validations: []validation{{Path: "validations[0]", Expression: tt.expression}}}
			result, err := e.EvalValidations(policy, input)
			if err != nil {
				t.Fatalf("EvalValidations() error: %v", err)
			}
			if !result.Allowed {
				t.Fatalf("expected Allowed=true; violations: %s", result.FormatViolations())
			}
		})
	}
}

func TestParseAdmissionInputFallsBackToUnstructured(t *testing.T) {
	input, err := ParseAdmissionInput(`
object:
  apiVersion: example.com/v1
  kind: Widget
  metadata:
    name: example
  spec:
    replicas: 4
`)
	if err != nil {
		t.Fatalf("ParseAdmissionInput() error: %v", err)
	}
	object, ok := input.object.(map[string]interface{})
	if !ok {
		t.Fatalf("object type = %T, want map[string]interface{}", input.object)
	}
	got := object["spec"].(map[string]interface{})["replicas"]
	if _, ok := got.(int64); !ok {
		t.Errorf("replicas type = %T, want int64", got)
	}

	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}
	policy := &AdmissionPolicy{validations: []validation{{Path: "validations[0]", Expression: "object.spec.replicas % 2 == 0"}}}
	result, err := e.EvalValidations(policy, input)
	if err != nil {
		t.Fatalf("EvalValidations() error: %v", err)
	}
	if !result.Allowed {
		t.Fatalf("expected Allowed=true; violations: %s", result.FormatViolations())
	}
}

func TestParseAdmissionInputUnknownMetaKindFallsBackToUnstructured(t *testing.T) {
	input, err := ParseAdmissionInput(`
object:
  apiVersion: example.com/v1
  kind: Status
  metadata:
    name: example
  spec:
    marker: kept
`)
	if err != nil {
		t.Fatalf("ParseAdmissionInput() error: %v", err)
	}
	object, ok := input.object.(map[string]interface{})
	if !ok {
		t.Fatalf("object type = %T, want map[string]interface{}", input.object)
	}
	got := object["spec"].(map[string]interface{})["marker"]
	if got != "kept" {
		t.Errorf("spec.marker = %v, want kept", got)
	}
}

func TestParseAdmissionInputRejectsInvalidRegisteredType(t *testing.T) {
	_, err := ParseAdmissionInput(`
object:
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: d
  spec:
    replicas: "4"
`)
	if err == nil {
		t.Fatal("expected invalid replicas type to be rejected")
	}
	if !strings.Contains(err.Error(), "replicas") {
		t.Fatalf("error = %q, want replicas field error", err)
	}
}

func TestUnstructuredInputPreservesCallerNumberType(t *testing.T) {
	u, err := convertObjectToUnstructured(map[string]interface{}{
		"spec": map[string]interface{}{"value": float64(4)},
	})
	if err != nil {
		t.Fatalf("convertObjectToUnstructured() error: %v", err)
	}
	got := u.Object["spec"].(map[string]interface{})["value"]
	if _, ok := got.(float64); !ok {
		t.Errorf("value type = %T, want float64", got)
	}
}
