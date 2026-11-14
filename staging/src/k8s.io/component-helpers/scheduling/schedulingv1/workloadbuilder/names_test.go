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

package workloadbuilder

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestGenerateWorkloadName(t *testing.T) {
	testCases := []struct {
		name string
		job  string
		uid  types.UID
		want func(t *testing.T, generated string)
	}{
		{
			name: "name is within the maximum length",
			job:  strings.Repeat("job", 100),
			uid:  types.UID("uid-1"),
			want: func(t *testing.T, generated string) {
				if len(generated) > validation.DNS1123SubdomainMaxLength {
					t.Fatalf("generated name is too long: %d", len(generated))
				}
			},
		},
		{
			name: "name is deterministic and preserves the existing format",
			job:  "job",
			uid:  types.UID("uid-1"),
			want: func(t *testing.T, generated string) {
				if generated != "job-647cf7d9" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
			},
		},
		{
			name: "empty owner name",
			uid:  types.UID("uid-1"),
			want: func(t *testing.T, generated string) {
				if generated != "647cf7d9" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
			},
		},
		{
			name: "different UIDs produce different names",
			job:  strings.Repeat("job", 100),
			uid:  types.UID("uid-1"),
			want: func(t *testing.T, generated string) {
				if generated == GenerateWorkloadName(strings.Repeat("job", 100), types.UID("uid-2")) {
					t.Fatal("different UIDs should produce different names")
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.want(t, GenerateWorkloadName(tc.job, tc.uid))
		})
	}
}

func TestGeneratePodGroupName(t *testing.T) {
	testCases := []struct {
		name     string
		workload string
		template string
		uid      types.UID
		want     func(t *testing.T, generated string)
	}{
		{
			name:     "name is within the maximum length",
			workload: strings.Repeat("workload", 40),
			template: strings.Repeat("template", 40),
			want: func(t *testing.T, generated string) {
				if len(generated) > validation.DNS1123SubdomainMaxLength {
					t.Fatalf("generated name is too long: %d", len(generated))
				}
			},
		},
		{
			name:     "short workload keeps its full name",
			workload: "workload",
			template: strings.Repeat("template", 40),
			want: func(t *testing.T, generated string) {
				if !strings.HasPrefix(generated, "workload-") {
					t.Fatalf("generated name should retain the workload name: %q", generated)
				}
				if len(generated) > validation.DNS1123SubdomainMaxLength {
					t.Fatalf("generated name is too long: %d", len(generated))
				}
			},
		},
		{
			name:     "short template keeps its full name",
			workload: strings.Repeat("workload", 40),
			template: "template",
			want: func(t *testing.T, generated string) {
				if !strings.Contains(generated, "-template-") {
					t.Fatalf("generated name should retain the template name: %q", generated)
				}
				if len(generated) > validation.DNS1123SubdomainMaxLength {
					t.Fatalf("generated name is too long: %d", len(generated))
				}
			},
		},
		{
			name:     "name is deterministic",
			workload: strings.Repeat("workload", 40),
			template: strings.Repeat("template", 40),
			want: func(t *testing.T, generated string) {
				if generated != GeneratePodGroupName(strings.Repeat("workload", 40), strings.Repeat("template", 40), "") {
					t.Fatal("generated name is not deterministic")
				}
			},
		},
		{
			name:     "different templates produce different names",
			workload: strings.Repeat("workload", 40),
			template: strings.Repeat("template", 40),
			want: func(t *testing.T, generated string) {
				if generated == GeneratePodGroupName(strings.Repeat("workload", 40), "other-template", "") {
					t.Fatal("different templates should produce different names")
				}
			},
		},
		{
			name:     "owner UID is part of the hash and the name is stable",
			workload: "workload",
			template: "template",
			uid:      "uid-1",
			want: func(t *testing.T, generated string) {
				if generated != "workload-template-57db668b94" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
				if generated == GeneratePodGroupName("workload", "template", "uid-2") {
					t.Fatal("different owner UIDs should produce different names")
				}
				if generated == GeneratePodGroupName("workload", "template", "") {
					t.Fatal("an owner UID should change the name")
				}
			},
		},
		{
			name:     "hash inputs are delimited",
			workload: "foo",
			template: "barbaz",
			uid:      "uid-1",
			want: func(t *testing.T, generated string) {
				if hashOf(generated) == hashOf(GeneratePodGroupName("foobar", "baz", "uid-1")) {
					t.Fatal("shifting characters between workload and template should change the hash")
				}
				if hashOf(generated) == hashOf(GeneratePodGroupName("foo", "barbazuid-1", "")) {
					t.Fatal("shifting characters between template and owner UID should change the hash")
				}
			},
		},
		{
			name:     "owner UID with long names stays within the maximum length",
			workload: strings.Repeat("workload", 40),
			template: strings.Repeat("template", 40),
			uid:      "uid-1",
			want: func(t *testing.T, generated string) {
				if len(generated) > validation.DNS1123SubdomainMaxLength {
					t.Fatalf("generated name is too long: %d", len(generated))
				}
			},
		},
		{
			name:     "both names empty returns only the hash",
			workload: "",
			template: "",
			want: func(t *testing.T, generated string) {
				if generated != "6f6fd8cd5" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
			},
		},
		{
			name:     "empty workload keeps the template name",
			workload: "",
			template: "template",
			want: func(t *testing.T, generated string) {
				if generated != "template-7865ccf4cf" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
			},
		},
		{
			name:     "empty template keeps the workload name",
			workload: "workload",
			template: "",
			want: func(t *testing.T, generated string) {
				if generated != "workload-76fc8fddcb" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
			},
		},
		{
			name:     "empty workload with long template stays within the maximum length",
			workload: "",
			template: strings.Repeat("template", 40),
			want: func(t *testing.T, generated string) {
				if len(generated) > validation.DNS1123SubdomainMaxLength {
					t.Fatalf("generated name is too long: %d", len(generated))
				}
			},
		},
		{
			name:     "empty template with long workload stays within the maximum length",
			workload: strings.Repeat("workload", 40),
			template: "",
			want: func(t *testing.T, generated string) {
				if len(generated) > validation.DNS1123SubdomainMaxLength {
					t.Fatalf("generated name is too long: %d", len(generated))
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.want(t, GeneratePodGroupName(tc.workload, tc.template, tc.uid))
		})
	}
}

// hashOf returns the trailing hash segment of a generated name so that
// collision checks compare hashes rather than the human-readable prefixes.
func hashOf(name string) string {
	return name[strings.LastIndex(name, "-")+1:]
}
