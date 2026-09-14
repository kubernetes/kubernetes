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
				if generated != GeneratePodGroupName(strings.Repeat("workload", 40), strings.Repeat("template", 40)) {
					t.Fatal("generated name is not deterministic")
				}
			},
		},
		{
			name:     "different templates produce different names",
			workload: strings.Repeat("workload", 40),
			template: strings.Repeat("template", 40),
			want: func(t *testing.T, generated string) {
				if generated == GeneratePodGroupName(strings.Repeat("workload", 40), "other-template") {
					t.Fatal("different templates should produce different names")
				}
			},
		},
		{
			name:     "both names empty returns only the hash",
			workload: "",
			template: "",
			want: func(t *testing.T, generated string) {
				if generated != "65bb57b6b5" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
			},
		},
		{
			name:     "empty workload keeps the template name",
			workload: "",
			template: "template",
			want: func(t *testing.T, generated string) {
				if generated != "template-5cbb944dc9" {
					t.Fatalf("unexpected generated name: %q", generated)
				}
			},
		},
		{
			name:     "empty template keeps the workload name",
			workload: "workload",
			template: "",
			want: func(t *testing.T, generated string) {
				if generated != "workload-c5546b9dd" {
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
			tc.want(t, GeneratePodGroupName(tc.workload, tc.template))
		})
	}
}
