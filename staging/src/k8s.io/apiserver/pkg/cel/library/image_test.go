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
	"regexp"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/cel/library"

	apiservercel "k8s.io/apiserver/pkg/cel"
)

func testImageLib(t *testing.T, expr string, expectResult ref.Val, expectRuntimeErrPattern string, expectCompileErrs []string) {
	env, err := cel.NewEnv(
		library.Image(),
	)
	if err != nil {
		t.Fatalf("%v", err)
	}
	compiled, issues := env.Compile(expr)

	if len(expectCompileErrs) > 0 {
		missingCompileErrs := []string{}
		matchedCompileErrs := sets.New[int]()
		for _, expectedCompileErr := range expectCompileErrs {
			compiledPattern, err := regexp.Compile(expectedCompileErr)
			if err != nil {
				t.Fatalf("failed to compile expected err regex: %v", err)
			}

			didMatch := false

			for i, compileError := range issues.Errors() {
				if compiledPattern.Match([]byte(compileError.Message)) {
					didMatch = true
					matchedCompileErrs.Insert(i)
				}
			}

			if !didMatch {
				missingCompileErrs = append(missingCompileErrs, expectedCompileErr)
			} else if len(matchedCompileErrs) != len(issues.Errors()) {
				unmatchedErrs := []cel.Error{}
				for i, issue := range issues.Errors() {
					if !matchedCompileErrs.Has(i) {
						unmatchedErrs = append(unmatchedErrs, *issue)
					}
				}
				require.Empty(t, unmatchedErrs, "unexpected compilation errors")
			}
		}

		require.Empty(t, missingCompileErrs, "expected compilation errors")
		return
	} else if len(issues.Errors()) > 0 {
		for _, err := range issues.Errors() {
			t.Errorf("unexpected compile error: %v", err)
		}
		t.FailNow()
	}

	prog, err := env.Program(compiled)
	if err != nil {
		t.Fatalf("%v", err)
	}
	res, _, err := prog.Eval(map[string]interface{}{})
	if len(expectRuntimeErrPattern) > 0 {
		if err == nil {
			t.Fatalf("no runtime error thrown. Expected: %v", expectRuntimeErrPattern)
		} else if matched, regexErr := regexp.MatchString(expectRuntimeErrPattern, err.Error()); regexErr != nil {
			t.Fatalf("failed to compile expected err regex: %v", regexErr)
		} else if !matched {
			t.Fatalf("unexpected err: %v", err)
		}
	} else if err != nil {
		t.Fatalf("%v", err)
	} else if expectResult != nil {
		converted := res.Equal(expectResult).Value().(bool)
		require.True(t, converted, "expectation not equal to output")
	} else {
		t.Fatal("expected result must not be nil")
	}
}

func TestImage(t *testing.T) {
	trueVal := types.Bool(true)
	falseVal := types.Bool(false)

	cases := []struct {
		name               string
		expr               string
		expectValue        ref.Val
		expectedCompileErr []string
		expectedRuntimeErr string
	}{
		{
			name: "parse",
			expr: `image("registry.k8s.io/kube-apiserver-arm64:latest")`,
			expectValue: apiservercel.Image{
				ImageReference: apiservercel.ImageReference{
					Image:      "registry.k8s.io/kube-apiserver-arm64:latest",
					Registry:   "registry.k8s.io",
					Repository: "kube-apiserver-arm64",
					Tag:        "latest",
					Identifier: "latest",
				},
			},
		},
		{
			name: "parse_normalized",
			expr: `image("kube-apiserver-arm64", true)`,
			expectValue: apiservercel.Image{
				ImageReference: apiservercel.ImageReference{
					Image:      "docker.io/library/kube-apiserver-arm64:latest",
					Registry:   "docker.io",
					Repository: "library/kube-apiserver-arm64",
					Tag:        "latest",
					Identifier: "latest",
				},
			},
		},
		{
			name: "parse_non_normalized",
			expr: `image("kube-apiserver-arm64", false)`,
			expectValue: apiservercel.Image{
				ImageReference: apiservercel.ImageReference{
					Image:      "kube-apiserver-arm64",
					Registry:   "",
					Repository: "kube-apiserver-arm64",
					Tag:        "",
					Identifier: "",
				},
			},
		},
		{
			name: "parse_normalized_not_required",
			expr: `image("registry.k8s.io/kube-apiserver-arm64:latest", true)`,
			expectValue: apiservercel.Image{
				ImageReference: apiservercel.ImageReference{
					Image:      "registry.k8s.io/kube-apiserver-arm64:latest",
					Registry:   "registry.k8s.io",
					Repository: "kube-apiserver-arm64",
					Tag:        "latest",
					Identifier: "latest",
				},
			},
		},
		{
			name: "parse_normalized_not_required_disabled",
			expr: `image("registry.k8s.io/kube-apiserver-arm64:latest", false)`,
			expectValue: apiservercel.Image{
				ImageReference: apiservercel.ImageReference{
					Image:      "registry.k8s.io/kube-apiserver-arm64:latest",
					Registry:   "registry.k8s.io",
					Repository: "kube-apiserver-arm64",
					Tag:        "latest",
					Identifier: "latest",
				},
			},
		},
		{
			name:               "parse_invalid_image",
			expr:               `image("registry.k8s.io/kube-apiserver-arm64:@")`,
			expectedRuntimeErr: "failed to parse image: registry.k8s.io/kube-apiserver-arm64:@, err: invalid reference format",
		},
		{
			name:        "isImage",
			expr:        `isImage("registry.k8s.io/kube-apiserver-arm64:latest")`,
			expectValue: trueVal,
		},
		{
			name:        "isImage_false",
			expr:        `isImage("registry.k8s.io/kube-apiserver-arm64:@")`,
			expectValue: falseVal,
		},
		{
			name:               "isImage_noOverload",
			expr:               `isImage(0)`,
			expectedCompileErr: []string{"found no matching overload for 'isImage' applied to.*"},
		},
		{
			name:        "contains_digest_no_identifier",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64").containsDigest()`,
			expectValue: falseVal,
		},
		{
			name:        "contains_digest_tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64:latest").containsDigest()`,
			expectValue: falseVal,
		},
		{
			name:        "contains_digest_true",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").containsDigest()`,
			expectValue: trueVal,
		},
		{
			name:        "contains_digest_with_tag_true",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").containsDigest()`,
			expectValue: trueVal,
		},
		{
			name:        "registry",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64").registry() == "registry.k8s.io"`,
			expectValue: trueVal,
		},
		{
			name:        "no_registry",
			expr:        `image("kube-apiserver-arm64:testtag").registry()`,
			expectValue: types.String(""),
		},
		{
			name:        "registry_normalized",
			expr:        `image("kube-apiserver-arm64:testtag", true).registry()`,
			expectValue: types.String("docker.io"),
		},
		{
			name:        "registry_matches",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64").registry().matches("(registry.k8s.io|ghcr.io)")`,
			expectValue: trueVal,
		},
		{
			name:        "repository",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64").repository() == "kube-apiserver-arm64"`,
			expectValue: trueVal,
		},
		{
			name:        "repository_normalized",
			expr:        `image("kube-apiserver-arm64:testtag", true).repository()`,
			expectValue: types.String("library/kube-apiserver-arm64"),
		},
		{
			name:        "empty_identifier",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64").identifier()`,
			expectValue: types.String(""),
		},
		{
			name:        "empty_identifier_not_normalized",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64", false).identifier()`,
			expectValue: types.String(""),
		},
		{
			name:        "empty_identifier_normalized",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64", true).identifier()`,
			expectValue: types.String("latest"),
		},
		{
			name:        "identifier_tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64:testtag").identifier()`,
			expectValue: types.String("testtag"),
		},
		{
			name:        "identifer_digest",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").identifier()`,
			expectValue: types.String("sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"),
		},
		{
			name:        "identifer_digest_and_tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").identifier()`,
			expectValue: types.String("sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"),
		},
		{
			name:        "tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64:testtag").tag()`,
			expectValue: types.String("testtag"),
		},
		{
			name:        "empty_tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64").tag()`,
			expectValue: types.String(""),
		},
		{
			name:        "empty_tag_not_normalized",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64", false).tag()`,
			expectValue: types.String(""),
		},
		{
			name:        "empty_tag_normalized",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64", true).tag()`,
			expectValue: types.String("latest"),
		},
		{
			name:        "no_tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").tag()`,
			expectValue: types.String(""),
		},
		{
			name:        "no_digest",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64").digest()`,
			expectValue: types.String(""),
		},
		{
			name:        "digest_tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64:testtag").digest()`,
			expectValue: types.String(""),
		},
		{
			name:        "digest",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").digest()`,
			expectValue: types.String("sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"),
		},
		{
			name:        "docker_ref_digest",
			expr:        `image("docker.io/library/nginx@sha256:ed6d2c43c8fbcd3eaa44c9dab6d94cb346234476230dc1681227aa72d07181ee").digest()`,
			expectValue: types.String("sha256:ed6d2c43c8fbcd3eaa44c9dab6d94cb346234476230dc1681227aa72d07181ee"),
		},
		{
			name:        "digest_digest_and_tag",
			expr:        `image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").digest() == "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"`,
			expectValue: trueVal,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			testImageLib(t, c.expr, c.expectValue, c.expectedRuntimeErr, c.expectedCompileErr)
		})
	}
}
