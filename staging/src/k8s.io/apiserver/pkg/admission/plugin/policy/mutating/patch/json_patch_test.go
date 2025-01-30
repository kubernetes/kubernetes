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

package patch

import (
	"context"
	"github.com/google/go-cmp/cmp"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/utils/ptr"
)

func TestJSONPatch(t *testing.T) {
	deploymentGVR := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	tests := []struct {
		name              string
		expression        string
		gvr               schema.GroupVersionResource
		object, oldObject runtime.Object
		expectedResult    runtime.Object
		expectedErr       string
	}{
		{
			name: "jsonPatch with false test operation",
			expression: `[
						JSONPatch{op: "test", path: "/spec/replicas", value: 100}, 
						JSONPatch{op: "replace", path: "/spec/replicas", value: 3},
					]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: ptr.To[int32](1)}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: ptr.To[int32](1)}},
		},
		{
			name: "jsonPatch with true test operation",
			expression: `[
						JSONPatch{op: "test", path: "/spec/replicas", value: 1}, 
						JSONPatch{op: "replace", path: "/spec/replicas", value: 3},
					]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: ptr.To[int32](1)}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: ptr.To[int32](3)}},
		},
		{
			name: "jsonPatch remove to unset field",
			expression: `[
					JSONPatch{op: "remove", path: "/spec/replicas"}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Replicas: ptr.To[int32](1)}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch remove map entry by key",
			expression: `[
					JSONPatch{op: "remove", path: "/metadata/labels/y"}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"x": "1", "y": "1"}}, Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"x": "1"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch remove element in list",
			expression: `[
					JSONPatch{op: "remove", path: "/spec/template/spec/containers/1"}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
			}}}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}, {Name: "c"}},
			}}}},
		},
		{
			name: "jsonPatch copy map entry by key",
			expression: `[
					JSONPatch{op: "copy", from: "/metadata/labels/x", path: "/metadata/labels/y"}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"x": "1"}}, Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"x": "1", "y": "1"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch copy first element to end of list",
			expression: `[
					JSONPatch{op: "copy", from: "/spec/template/spec/containers/0", path: "/spec/template/spec/containers/-"}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
			}}}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}, {Name: "a"}},
			}}}},
		},
		{
			name: "jsonPatch move map entry by key",
			expression: `[
					JSONPatch{op: "move", from: "/metadata/labels/x", path: "/metadata/labels/y"}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"x": "1"}}, Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "1"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch move first element to end of list",
			expression: `[
					JSONPatch{op: "move", from: "/spec/template/spec/containers/0", path: "/spec/template/spec/containers/-"}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
			}}}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "b"}, {Name: "c"}, {Name: "a"}},
			}}}},
		},
		{
			name: "jsonPatch add map entry by key and value",
			expression: `[
					JSONPatch{op: "add", path: "/metadata/labels/x", value: "2"}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "1"}}, Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "1", "x": "2"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch add map value to field",
			expression: `[
					JSONPatch{op: "add", path: "/metadata/labels", value: {"y": "2"}}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "2"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch add map to existing map", // performs a replacement
			expression: `[
					JSONPatch{op: "add", path: "/metadata/labels", value: {"y": "2"}}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"x": "1"}}, Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "2"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch add to start of list",
			expression: `[
					JSONPatch{op: "add", path: "/spec/template/spec/containers/0", value: {"name": "x"}}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}},
			}}}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "x"}, {Name: "a"}},
			}}}},
		},
		{
			name: "jsonPatch add to end of list",
			expression: `[
					JSONPatch{op: "add", path: "/spec/template/spec/containers/-", value: {"name": "x"}}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}},
			}}}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}, {Name: "x"}},
			}}}},
		},
		{
			name: "jsonPatch replace key in map",
			expression: `[
					JSONPatch{op: "replace", path: "/metadata/labels/x", value: "2"}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "1"}}, Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "1", "x": "2"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch replace map value of unset field", // adds the field value
			expression: `[
					JSONPatch{op: "replace", path: "/metadata/labels", value: {"y": "2"}}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "2"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch replace map value of set field",
			expression: `[
					JSONPatch{op: "replace", path: "/metadata/labels", value: {"y": "2"}}, 
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"x": "1"}}, Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"y": "2"}}, Spec: appsv1.DeploymentSpec{}},
		},
		{
			name: "jsonPatch replace first element in list",
			expression: `[
					JSONPatch{op: "replace", path: "/spec/template/spec/containers/0", value: {"name": "x"}}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}},
			}}}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "x"}},
			}}}},
		},
		{
			name: "jsonPatch add map entry by key and value",
			expression: `[
					JSONPatch{op: "add", path: "/spec", value: Object.spec{selector: Object.spec.selector{}, replicas: 10}}
				]`,
			gvr:            deploymentGVR,
			object:         &appsv1.Deployment{Spec: appsv1.DeploymentSpec{}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Selector: &metav1.LabelSelector{}, Replicas: ptr.To[int32](10)}},
		},
		{
			name: "JSONPatch patch type has field access",
			expression: `[
					JSONPatch{
						op: "add", path: "/metadata/labels",
						value: {
							"op": JSONPatch{op: "opValue"}.op,
							"path": JSONPatch{path: "pathValue"}.path,
							"from": JSONPatch{from: "fromValue"}.from,
							"value": string(JSONPatch{value: "valueValue"}.value),
						}
					}
				]`,
			gvr:    deploymentGVR,
			object: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				"op":    "opValue",
				"path":  "pathValue",
				"from":  "fromValue",
				"value": "valueValue",
			}}},
		},
		{
			name: "JSONPatch patch type has field testing",
			expression: `[
					JSONPatch{
						op: "add", path: "/metadata/labels",
						value: {
							"op": string(has(JSONPatch{op: "opValue"}.op)),
							"path": string(has(JSONPatch{path: "pathValue"}.path)),
							"from": string(has(JSONPatch{from: "fromValue"}.from)),
							"value": string(has(JSONPatch{value: "valueValue"}.value)),
							"op-unset": string(has(JSONPatch{}.op)),
							"path-unset": string(has(JSONPatch{}.path)),
							"from-unset": string(has(JSONPatch{}.from)),
							"value-unset": string(has(JSONPatch{}.value)),
						}
					}
				]`,
			gvr:    deploymentGVR,
			object: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				"op":          "true",
				"path":        "true",
				"from":        "true",
				"value":       "true",
				"op-unset":    "false",
				"path-unset":  "false",
				"from-unset":  "false",
				"value-unset": "false",
			}}},
		},
		{
			name: "JSONPatch patch type equality",
			expression: `[
					JSONPatch{
						op: "add", path: "/metadata/labels",
						value: {
							"empty": string(JSONPatch{} == JSONPatch{}),
							"partial": string(JSONPatch{op: "add"} == JSONPatch{op: "add"}),
							"same-all": string(JSONPatch{op: "add", path: "path", from: "from", value: 1} == JSONPatch{op: "add", path: "path", from: "from", value: 1}),
							"different-op": string(JSONPatch{op: "add"} == JSONPatch{op: "remove"}),
							"different-path": string(JSONPatch{op: "add", path: "x", from: "from", value: 1} == JSONPatch{op: "add", path: "path", from: "from", value: 1}),
							"different-from": string(JSONPatch{op: "add", path: "path", from: "x", value: 1} == JSONPatch{op: "add", path: "path", from: "from", value: 1}),
							"different-value": string(JSONPatch{op: "add", path: "path", from: "from", value: "1"} == JSONPatch{op: "add", path: "path", from: "from", value: 1}),
						}
					}
				]`,
			gvr:    deploymentGVR,
			object: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				"empty":           "true",
				"partial":         "true",
				"same-all":        "true",
				"different-op":    "false",
				"different-path":  "false",
				"different-from":  "false",
				"different-value": "false",
			}}},
		},
		{
			name: "JSONPatch key escaping",
			expression: `[
					JSONPatch{
						op: "add", path: "/metadata/labels", value: {}
					},
					JSONPatch{
						op: "add", path: "/metadata/labels/" + jsonpatch.escapeKey("k8s.io/x~y"), value: "true"
					}
				]`,
			gvr:    deploymentGVR,
			object: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}},
			expectedResult: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				"k8s.io/x~y": "true",
			}}},
		},
		{
			name: "jsonPatch with CEL initializer",
			expression: `[
					JSONPatch{op: "add", path: "/spec/template/spec/containers/-", value: Object.spec.template.spec.containers{
							name: "x",
							ports: [Object.spec.template.spec.containers.ports{containerPort: 8080}],
						}
					}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}},
			}}}},
			expectedResult: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}, {Name: "x", Ports: []corev1.ContainerPort{{ContainerPort: 8080}}}},
			}}}},
		},
		{
			name: "jsonPatch invalid CEL initializer field",
			expression: `[
					JSONPatch{
						op: "add", path: "/spec/template/spec/containers/-", 
						value: Object.spec.template.spec.containers{
							name: "x",
							ports: [Object.spec.template.spec.containers.ports{containerPortZ: 8080}]
						}
					}
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}},
			}}}},
			expectedErr: "strict decoding error: unknown field \"spec.template.spec.containers[1].ports[0].containerPortZ\"",
		},
		{
			name: "jsonPatch invalid CEL initializer type",
			expression: `[
					JSONPatch{
						op: "add", path: "/spec/template/spec/containers/-", 
						value: Object.spec.template.spec.containers{
							name: "x",
							ports: [Object.spec.template.spec.containers.portsZ{containerPort: 8080}]
						}
					}
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}},
			}}}},
			expectedErr: " mismatch: unexpected type name \"Object.spec.template.spec.containers.portsZ\", expected \"Object.spec.template.spec.containers.ports\", which matches field name path from root Object type",
		},
		{
			name: "jsonPatch replace end of list with - not allowed",
			expression: `[
					JSONPatch{op: "replace", path: "/spec/template/spec/containers/-", value: {"name": "x"}}, 
				]`,
			gvr: deploymentGVR,
			object: &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "a"}},
			}}}},
			expectedErr: "JSON Patch: replace operation does not apply: doc is missing key: /spec/template/spec/containers/-: missing value",
		},
	}

	compiler, err := cel.NewCompositedCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true))

	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			accessor := &JSONPatchCondition{Expression: tc.expression}
			compileResult := compiler.CompileMutatingEvaluator(accessor, cel.OptionalVariableDeclarations{StrictCost: true, HasPatchTypes: true}, environment.StoredExpressions)

			patcher := jsonPatcher{PatchEvaluator: compileResult}

			scheme := runtime.NewScheme()
			err := appsv1.AddToScheme(scheme)
			if err != nil {
				t.Fatal(err)
			}

			var gvk schema.GroupVersionKind
			gvks, _, err := scheme.ObjectKinds(tc.object)
			if err != nil {
				t.Fatal(err)
			}
			if len(gvks) == 1 {
				gvk = gvks[0]
			} else {
				t.Fatalf("Failed to find gvk for type: %T", tc.object)
			}

			metaAccessor, err := meta.Accessor(tc.object)
			if err != nil {
				t.Fatal(err)
			}

			attrs := admission.NewAttributesRecord(tc.object, tc.oldObject, gvk,
				metaAccessor.GetNamespace(), metaAccessor.GetName(), tc.gvr,
				"", admission.Create, &metav1.CreateOptions{}, false, nil)
			vAttrs := &admission.VersionedAttributes{
				Attributes:         attrs,
				VersionedKind:      gvk,
				VersionedObject:    tc.object,
				VersionedOldObject: tc.oldObject,
			}

			r := Request{
				MatchedResource:     tc.gvr,
				VersionedAttributes: vAttrs,
				ObjectInterfaces:    admission.NewObjectInterfacesFromScheme(scheme),
				OptionalVariables:   cel.OptionalVariableBindings{},
			}

			got, err := patcher.Patch(context.Background(), r, celconfig.RuntimeCELCostBudget)
			if len(tc.expectedErr) > 0 {
				if err == nil {
					t.Fatalf("expected error: %s", tc.expectedErr)
				} else {
					if !strings.Contains(err.Error(), tc.expectedErr) {
						t.Fatalf("expected error: %s, got: %s", tc.expectedErr, err.Error())
					}
					return
				}
			}
			if err != nil && len(tc.expectedErr) == 0 {
				t.Fatalf("unexpected error: %v", err)
			}
			if !equality.Semantic.DeepEqual(tc.expectedResult, got) {
				t.Errorf("unexpected result, got diff:\n%s\n", cmp.Diff(tc.expectedResult, got))
			}
		})
	}
}
