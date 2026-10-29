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

// Benchmarks in this file only use Kubernetes APIs (no direct cel-go imports)
// so that the same file builds against different cel-go versions and can be
// used for before/after comparisons of cel-go upgrades.

import (
	"context"
	"fmt"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
)

// benchRealisticPolicy is a ValidatingAdmissionPolicy shaped like policies seen
// in practice: composition variables, several validations iterating over the
// pod template's containers, a messageExpression, a regex check and an audit
// annotation.
func benchRealisticPolicy() *Policy {
	fail := admissionregistrationv1.Fail
	return &Policy{
		ObjectMeta: metav1.ObjectMeta{Name: "bench-realistic"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: &fail,
			Variables: []admissionregistrationv1.Variable{
				{Name: "containers", Expression: "object.spec.template.spec.containers"},
				{Name: "images", Expression: "variables.containers.map(c, c.image)"},
			},
			Validations: []admissionregistrationv1.Validation{
				{
					Expression:        "object.spec.replicas <= 50",
					MessageExpression: "'replicas ' + string(object.spec.replicas) + ' exceeds 50'",
				},
				{Expression: "variables.containers.all(c, c.image.startsWith('registry.example.com/'))"},
				{Expression: "variables.containers.all(c, has(c.resources) && has(c.resources.limits) && 'memory' in c.resources.limits)"},
				{Expression: "!variables.images.exists(i, i.endsWith(':latest'))"},
				{Expression: "object.metadata.name.matches('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')"},
			},
			AuditAnnotations: []admissionregistrationv1.AuditAnnotation{
				{Key: "container-count", ValueExpression: "string(size(variables.containers))"},
			},
		},
	}
}

func benchDeployment(containers int) *unstructured.Unstructured {
	cs := make([]interface{}, containers)
	for i := range containers {
		cs[i] = map[string]interface{}{
			"name":  fmt.Sprintf("c-%d", i),
			"image": fmt.Sprintf("registry.example.com/app-%d:v1", i),
			"resources": map[string]interface{}{
				"limits": map[string]interface{}{"memory": "128Mi", "cpu": "100m"},
			},
		}
	}
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",
		"metadata":   map[string]interface{}{"name": "bench-deploy", "namespace": "default"},
		"spec": map[string]interface{}{
			"replicas": int64(3),
			"template": map[string]interface{}{
				"spec": map[string]interface{}{"containers": cs},
			},
		},
	}}
}

var (
	benchDeploymentGVK = schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	benchDeploymentGVR = schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
)

// benchVersionedAttributes builds fresh request attributes, as a real admission
// request would. The CEL representation of the object is cached on the
// attributes, so building them per iteration keeps object conversion in the
// measurement, like it is for every request.
func benchVersionedAttributes(obj *unstructured.Unstructured) *admission.VersionedAttributes {
	attrs := admission.NewAttributesRecord(obj, nil, benchDeploymentGVK, "default", "bench-deploy",
		benchDeploymentGVR, "", admission.Create, &metav1.CreateOptions{}, false, nil)
	return &admission.VersionedAttributes{
		Attributes:      attrs,
		VersionedKind:   benchDeploymentGVK,
		VersionedObject: admission.NewLazyObject(obj),
	}
}

// BenchmarkVAPRealisticPolicyCompile measures compiling a whole policy
// (variables, validations, message expressions, audit annotations).
func BenchmarkVAPRealisticPolicyCompile(b *testing.B) {
	policy := benchRealisticPolicy()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = compilePolicy(policy)
	}
}

// BenchmarkVAPRealisticPolicyValidate measures evaluating a compiled policy
// against one admission request, scaling the number of containers iterated by
// the comprehensions.
func BenchmarkVAPRealisticPolicyValidate(b *testing.B) {
	v := compilePolicy(benchRealisticPolicy())
	ctx := context.TODO()
	for _, n := range []int{1, 10, 100} {
		b.Run(fmt.Sprintf("containers=%d", n), func(b *testing.B) {
			obj := benchDeployment(n)
			res := v.Validate(ctx, benchDeploymentGVR, benchVersionedAttributes(obj), nil, nil, celconfig.RuntimeCELCostBudget, nil)
			if len(res.Decisions) == 0 {
				b.Fatal("expected policy decisions")
			}
			for _, d := range res.Decisions {
				if d.Action != ActionAdmit {
					b.Fatalf("expected admit, got %v: %s", d.Action, d.Message)
				}
			}
			if len(res.AuditAnnotations) != 1 {
				b.Fatalf("expected 1 audit annotation, got %d", len(res.AuditAnnotations))
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = v.Validate(ctx, benchDeploymentGVR, benchVersionedAttributes(obj), nil, nil, celconfig.RuntimeCELCostBudget, nil)
			}
		})
	}
}
