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

package library_test

import (
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	"k8s.io/apiserver/pkg/cel/library"
)

func testPod(t *testing.T, expr string, activation map[string]interface{}, expectResult ref.Val, expectRuntimeErr string, expectCompileErrs []string) {
	t.Helper()
	env, err := cel.NewEnv(
		library.Pod(),
		cel.Variable("spec", cel.DynType),
	)
	if err != nil {
		t.Fatalf("unexpected error creating CEL env: %v", err)
	}

	compiled, issues := env.Compile(expr)
	if len(expectCompileErrs) > 0 {
		for _, expected := range expectCompileErrs {
			found := false
			for _, issue := range issues.Errors() {
				if issue.Message == expected {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expected compile error %q not found in %v", expected, issues.Errors())
			}
		}
		return
	}
	if len(issues.Errors()) > 0 {
		t.Fatalf("unexpected compile errors: %v", issues.Errors())
	}

	prog, err := env.Program(compiled)
	if err != nil {
		t.Fatalf("unexpected error creating program: %v", err)
	}

	res, _, err := prog.Eval(activation)
	if expectRuntimeErr != "" {
		if err == nil {
			t.Fatalf("expected runtime error %q but got none", expectRuntimeErr)
		}
		if err.Error() != expectRuntimeErr {
			t.Fatalf("expected runtime error %q but got %q", expectRuntimeErr, err.Error())
		}
		return
	}
	if err != nil {
		t.Fatalf("unexpected runtime error: %v", err)
	}
	if expectResult == nil {
		t.Fatal("expectResult must not be nil")
	}
	if res.Equal(expectResult) != types.True {
		t.Errorf("expected %v but got %v", expectResult, res)
	}
}

func makeContainer(name, image string) map[string]interface{} {
	return map[string]interface{}{"name": name, "image": image}
}

func TestAllContainers(t *testing.T) {
	cases := []struct {
		name             string
		expr             string
		spec             map[string]interface{}
		expectResult     ref.Val
		expectRuntimeErr string
	}{
		{
			name: "all three container types",
			expr: "allContainers(spec).size()",
			spec: map[string]interface{}{
				"initContainers":     []interface{}{makeContainer("init", "busybox")},
				"containers":         []interface{}{makeContainer("main", "nginx"), makeContainer("sidecar", "envoy")},
				"ephemeralContainers": []interface{}{makeContainer("debug", "alpine")},
			},
			expectResult: types.Int(4),
		},
		{
			name: "only regular containers",
			expr: "allContainers(spec).size()",
			spec: map[string]interface{}{
				"containers": []interface{}{makeContainer("main", "nginx")},
			},
			expectResult: types.Int(1),
		},
		{
			name: "no containers at all",
			expr: "allContainers(spec).size()",
			spec: map[string]interface{}{},
			expectResult: types.Int(0),
		},
		{
			name: "order is init then containers then ephemeral",
			expr: "allContainers(spec)[0].name == 'init' && allContainers(spec)[1].name == 'main' && allContainers(spec)[2].name == 'debug'",
			spec: map[string]interface{}{
				"initContainers":     []interface{}{makeContainer("init", "busybox")},
				"containers":         []interface{}{makeContainer("main", "nginx")},
				"ephemeralContainers": []interface{}{makeContainer("debug", "alpine")},
			},
			expectResult: types.True,
		},
		{
			name: "all() across all container types",
			expr: "allContainers(spec).all(c, c.image.startsWith('myregistry.io/'))",
			spec: map[string]interface{}{
				"initContainers": []interface{}{makeContainer("init", "myregistry.io/busybox")},
				"containers":     []interface{}{makeContainer("main", "myregistry.io/nginx")},
			},
			expectResult: types.True,
		},
		{
			name: "all() fails when one image doesn't match",
			expr: "allContainers(spec).all(c, c.image.startsWith('myregistry.io/'))",
			spec: map[string]interface{}{
				"containers": []interface{}{makeContainer("main", "docker.io/nginx")},
			},
			expectResult: types.False,
		},
		{
			name: "empty init and ephemeral containers",
			expr: "allContainers(spec).size()",
			spec: map[string]interface{}{
				"initContainers":     []interface{}{},
				"containers":         []interface{}{makeContainer("main", "nginx")},
				"ephemeralContainers": []interface{}{},
			},
			expectResult: types.Int(1),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testPod(t, tc.expr,
				map[string]interface{}{"spec": tc.spec},
				tc.expectResult, tc.expectRuntimeErr, nil)
		})
	}
}
