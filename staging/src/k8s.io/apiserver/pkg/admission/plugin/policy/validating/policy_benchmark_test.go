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
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/environment"
	celopenapi "k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/utils/ptr"
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

func benchTypedDeployment(containers int) *appsv1.Deployment {
	cs := make([]corev1.Container, containers)
	for i := range containers {
		cs[i] = corev1.Container{
			Name:  fmt.Sprintf("c-%d", i),
			Image: fmt.Sprintf("registry.example.com/app-%d:v1", i),
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{
					corev1.ResourceMemory: resource.MustParse("128Mi"),
					corev1.ResourceCPU:    resource.MustParse("100m"),
				},
			},
		}
	}
	return &appsv1.Deployment{
		TypeMeta:   metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{Name: "bench-deploy", Namespace: "default"},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To[int32](3),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{Containers: cs},
			},
		},
	}
}

type benchOpenAPIDiscovery struct {
	discovery.DiscoveryInterface
}

func (benchOpenAPIDiscovery) OpenAPIV3() openapi.Client {
	return openapitest.NewEmbeddedFileClient()
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

// BenchmarkVAPTypedToVal measures evaluating VAP CEL expressions against a
// typed Deployment wrapped with common.TypedToVal and its OpenAPI schema,
// scaling the number of containers iterated by the comprehensions.
func BenchmarkVAPTypedToVal(b *testing.B) {
	rawSchema, err := (&resolver.ClientDiscoveryResolver{Discovery: benchOpenAPIDiscovery{}}).ResolveSchema(benchDeploymentGVK)
	if err != nil {
		b.Fatalf("failed to resolve Deployment schema: %v", err)
	}
	deploySchema := &celopenapi.Schema{Schema: rawSchema}

	compiler, err := cel.NewCompositedCompiler(getCompositionEnvTemplateWithStrictCost())
	if err != nil {
		b.Fatalf("failed to create compiler: %v", err)
	}
	optionalVars := cel.OptionalVariableDeclarations{HasParams: false, HasAuthorizer: false}
	policy := benchRealisticPolicy()
	compiler.CompileAndStoreVariables(convertv1beta1Variables(policy.Spec.Variables), optionalVars, environment.StoredExpressions)

	exprs := []cel.ExpressionAccessor{
		&ValidationCondition{Expression: "object.spec.replicas <= 50"},
		&ValidationCondition{Expression: "variables.containers.all(c, c.image.startsWith('registry.example.com/'))"},
		&ValidationCondition{Expression: "variables.containers.all(c, has(c.resources) && has(c.resources.limits) && size(c.resources.limits) > 0)"},
		&ValidationCondition{Expression: "!variables.images.exists(i, i.endsWith(':latest'))"},
		&ValidationCondition{Expression: "object.metadata.name.matches('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')"},
	}
	compiled := make([]cel.CompilationResult, len(exprs))
	for i, expr := range exprs {
		compiled[i] = compiler.CompileCELExpression(expr, optionalVars, environment.StoredExpressions)
		if compiled[i].Error != nil {
			b.Fatalf("compile error for %q: %v", expr.GetExpression(), compiled[i].Error)
		}
	}

	ctx := context.TODO()
	for _, n := range []int{1, 10, 100} {
		b.Run(fmt.Sprintf("containers=%d", n), func(b *testing.B) {
			obj := benchTypedDeployment(n)
			eval := func() {
				compCtx := compiler.CreateContext(ctx)
				act := map[string]any{
					"object": common.TypedToVal(obj, deploySchema),
				}
				act["variables"] = compCtx.Variables(act)
				for _, c := range compiled {
					val, _, err := c.Program.ContextEval(compCtx, act)
					if err != nil || val.Value() != true {
						b.Fatalf("unexpected eval result: %v, err: %v", val, err)
					}
				}
			}
			eval()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				eval()
			}
		})
	}
}
