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

import "testing"

// Integers in unstructured input (from YAML or a hand-built map) must reach CEL
// as int64 so integer operations like modulo work instead of failing with "no
// such overload". convertObjectToUnstructured normalizes the whole-number
// float64 values that a plain JSON/YAML decode produces.
func TestUnstructuredIntegersWorkInCEL(t *testing.T) {
	e, err := NewEvaluator()
	if err != nil {
		t.Fatalf("NewEvaluator() error: %v", err)
	}

	yamlInput, err := ParseAdmissionInput(`
object:
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: d
  spec:
    replicas: 4
params:
  threshold: 2
`)
	if err != nil {
		t.Fatalf("ParseAdmissionInput() error: %v", err)
	}

	manualInput := NewAdmissionInput().SetUnstructuredObject(map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",
		"metadata":   map[string]interface{}{"name": "d"},
		"spec":       map[string]interface{}{"replicas": float64(4)},
	})

	tests := []struct {
		name       string
		input      *AdmissionInput
		expression string
		want       bool
	}{
		{name: "yaml modulo", input: yamlInput, expression: "object.spec.replicas % 2 == 0", want: true},
		{name: "yaml comparison", input: yamlInput, expression: "object.spec.replicas > 3", want: true},
		{name: "yaml modulo against params", input: yamlInput, expression: "object.spec.replicas % params.threshold == 0", want: true},
		{name: "hand-built float64 map modulo", input: manualInput, expression: "object.spec.replicas % 2 == 0", want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policy := &AdmissionPolicy{validations: []validation{{Path: "validations[0]", Expression: tt.expression}}}
			result, err := e.EvalValidations(policy, tt.input)
			if err != nil {
				t.Fatalf("EvalValidations() error: %v", err)
			}
			if result.Allowed != tt.want {
				t.Fatalf("Allowed = %v, want %v; violations: %s", result.Allowed, tt.want, result.FormatViolations())
			}
		})
	}
}

func TestConvertObjectToUnstructuredNormalizesNumbers(t *testing.T) {
	u, err := convertObjectToUnstructured(map[string]interface{}{
		"spec": map[string]interface{}{"replicas": float64(4)},
	})
	if err != nil {
		t.Fatalf("convertObjectToUnstructured() error: %v", err)
	}
	got := u.Object["spec"].(map[string]interface{})["replicas"]
	if _, ok := got.(int64); !ok {
		t.Errorf("replicas type = %T, want int64", got)
	}
}
