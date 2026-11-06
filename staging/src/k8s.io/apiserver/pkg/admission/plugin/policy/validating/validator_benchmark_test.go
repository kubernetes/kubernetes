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

package validating

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

func BenchmarkVAPConditionCompilation(b *testing.B) {
	fc := cel.NewConditionCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))
	exprs := []cel.ExpressionAccessor{
		&ValidationCondition{Expression: "object.spec.replicas > 0 && object.spec.template.spec.containers.all(c, c.image != '')"},
	}
	vars := cel.OptionalVariableDeclarations{HasParams: false, HasAuthorizer: false}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fc.CompileCondition(exprs, vars, environment.StoredExpressions)
	}
}

func BenchmarkVAPConditionEvaluation(b *testing.B) {
	fc := cel.NewConditionCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))
	exprs := []cel.ExpressionAccessor{
		&ValidationCondition{Expression: "object.spec.replicas > 0"},
	}
	vars := cel.OptionalVariableDeclarations{HasParams: false, HasAuthorizer: false}
	cond := fc.CompileCondition(exprs, vars, environment.StoredExpressions)

	fakeAttr := admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "default", "foo", schema.GroupVersionResource{}, "", admission.Create, nil, false, nil)
	fakeVersionedAttr, _ := admission.NewVersionedAttributes(fakeAttr, schema.GroupVersionKind{}, nil)
	admissionRequest := cel.CreateAdmissionRequest(fakeVersionedAttr.Attributes, metav1.GroupVersionResource{}, metav1.GroupVersionKind{})
	optionalVars := cel.OptionalVariableBindings{}

	ctx := context.TODO()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = cond.ForInput(ctx, fakeVersionedAttr, admissionRequest, optionalVars, nil, 1000)
	}
}
