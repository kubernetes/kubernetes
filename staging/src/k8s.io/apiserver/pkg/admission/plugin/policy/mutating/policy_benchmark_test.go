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

package mutating

// Benchmarks in this file only use Kubernetes APIs (no direct cel-go imports)
// so that the same file builds against different cel-go versions and can be
// used for before/after comparisons of cel-go upgrades.

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/utils/ptr"
)

var (
	benchMAPDeploymentGVK = schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	benchMAPDeploymentGVR = schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
)

// benchMAPPolicies returns MutatingAdmissionPolicies that touch every container
// in the pod template, using each of the two mutation types.
func benchMAPPolicies() map[string]*v1.MutatingAdmissionPolicy {
	return map[string]*v1.MutatingAdmissionPolicy{
		// Server-side-apply style mutation: builds typed objects in CEL, then
		// merges them into the request object.
		"applyConfiguration": applyConfigurations(policy("bench-apply"), `Object{
			metadata: Object.metadata{labels: {"mutated-by": "map-bench"}},
			spec: Object.spec{template: Object.spec.template{spec: Object.spec.template.spec{
				containers: object.spec.template.spec.containers.map(c,
					Object.spec.template.spec.containers{name: c.name, imagePullPolicy: "IfNotPresent"})
			}}}
		}`),
		// JSON patch mutation: builds the patch operations in CEL.
		"jsonPatch": jsonPatches(policy("bench-jsonpatch"), v1.JSONPatch{Expression: `[
			JSONPatch{op: "add", path: "/metadata/labels/mutated-by", value: "map-bench"},
			JSONPatch{op: "replace", path: "/spec/template/spec/containers",
				value: object.spec.template.spec.containers.map(c,
					Object.spec.template.spec.containers{name: c.name, image: c.image, imagePullPolicy: "IfNotPresent"})}
		]`}),
	}
}

func benchMAPDeployment(containers int) *appsv1.Deployment {
	cs := make([]corev1.Container, containers)
	for i := range containers {
		cs[i] = corev1.Container{
			Name:  fmt.Sprintf("c-%d", i),
			Image: fmt.Sprintf("registry.example.com/app-%d:v1", i),
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("128Mi")},
			},
		}
	}
	return &appsv1.Deployment{
		TypeMeta:   metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{Name: "bench-deploy", Namespace: "default", Labels: map[string]string{"app": "bench"}},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To[int32](3),
			Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: cs}},
		},
	}
}

// BenchmarkMAPCompile measures compiling a MutatingAdmissionPolicy.
func BenchmarkMAPCompile(b *testing.B) {
	for name, p := range benchMAPPolicies() {
		b.Run(name, func(b *testing.B) {
			if ev := compilePolicy(p); ev.Error != nil {
				b.Fatalf("unexpected compile error: %v", ev.Error)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = compilePolicy(p)
			}
		})
	}
}

// BenchmarkMAPPatch measures applying a compiled MutatingAdmissionPolicy to one
// admission request: CEL evaluation of the mutation plus applying the result.
func BenchmarkMAPPatch(b *testing.B) {
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		b.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	b.Cleanup(cancel)
	tcManager := patch.NewTypeConverterManager(nil, openapitest.NewEmbeddedFileClient())
	go tcManager.Run(ctx)
	if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
		return tcManager.GetTypeConverter(benchMAPDeploymentGVK) != nil, nil
	}); err != nil {
		b.Fatal(err)
	}
	typeConverter := tcManager.GetTypeConverter(benchMAPDeploymentGVK)
	objectInterfaces := admission.NewObjectInterfacesFromScheme(scheme)

	for name, p := range benchMAPPolicies() {
		ev := compilePolicy(p)
		if ev.Error != nil {
			b.Fatalf("unexpected compile error: %v", ev.Error)
		}
		if len(ev.Mutators) != 1 {
			b.Fatalf("expected 1 mutator, got %d", len(ev.Mutators))
		}
		patcher := ev.Mutators[0]
		for _, n := range []int{1, 10, 100} {
			b.Run(fmt.Sprintf("%s/containers=%d", name, n), func(b *testing.B) {
				obj := benchMAPDeployment(n)
				request := func() patch.Request {
					attrs := admission.NewAttributesRecord(obj, nil, benchMAPDeploymentGVK, "default", "bench-deploy",
						benchMAPDeploymentGVR, "", admission.Create, &metav1.CreateOptions{}, false, nil)
					return patch.Request{
						MatchedResource: benchMAPDeploymentGVR,
						VersionedAttributes: &admission.VersionedAttributes{
							Attributes:      attrs,
							VersionedKind:   benchMAPDeploymentGVK,
							VersionedObject: admission.NewLazyObject(obj),
						},
						ObjectInterfaces:  objectInterfaces,
						OptionalVariables: cel.OptionalVariableBindings{Authorizer: fakeAuthorizer{}},
						TypeConverter:     typeConverter,
					}
				}
				out, err := patcher.Patch(ctx, request(), celconfig.RuntimeCELCostBudget)
				if err != nil {
					b.Fatalf("unexpected patch error: %v", err)
				}
				// ApplyConfiguration returns an unstructured object and JSONPatch a
				// typed one, so check the result in unstructured form.
				got, err := runtime.DefaultUnstructuredConverter.ToUnstructured(out)
				if err != nil {
					b.Fatal(err)
				}
				label, _, _ := unstructured.NestedString(got, "metadata", "labels", "mutated-by")
				containers, _, _ := unstructured.NestedSlice(got, "spec", "template", "spec", "containers")
				if label != "map-bench" || len(containers) != n {
					b.Fatalf("mutation was not applied as expected: label=%q containers=%d", label, len(containers))
				}
				if pullPolicy, _ := containers[n-1].(map[string]interface{})["imagePullPolicy"].(string); pullPolicy != string(corev1.PullIfNotPresent) {
					b.Fatalf("expected imagePullPolicy IfNotPresent on last container, got %q", pullPolicy)
				}
				if obj.Labels["mutated-by"] != "" {
					b.Fatal("input object was modified in place")
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = patcher.Patch(ctx, request(), celconfig.RuntimeCELCostBudget)
				}
			})
		}
	}
}
