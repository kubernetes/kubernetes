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

package tableconvertor

import (
	"context"
	"testing"
	"time"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const Wide metav1.IncludeObjectPolicy = "wide"

func TestPrintCRD(t *testing.T) {
	tests := []struct {
		name   string
		crd    *apiextensions.CustomResourceDefinition
		expect []any
	}{
		{
			name: "Return table for CRD with wide options",
			crd: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foos.sample.example.com",
					CreationTimestamp: metav1.NewTime(time.Date(2025, 5, 3, 16, 36, 32, 0, time.UTC)),
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "sample.example.com",
					Scope: apiextensions.NamespaceScoped,
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "v1", Served: true, Storage: true},
						{Name: "v1beta1", Served: true, Storage: false},
					},
					Names: apiextensions.CustomResourceDefinitionNames{Kind: "Foo", ShortNames: []string{"f"}},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					Conditions: []apiextensions.CustomResourceDefinitionCondition{
						{Type: apiextensions.Established, Status: apiextensions.ConditionTrue},
					},
				},
			},
			expect: []any{
				"foos.sample.example.com",
				"Namespaced",
				"v1(storage),v1beta1",
				"2025-05-03T16:36:32Z",
				"sample.example.com",
				"Foo",
				"f",
				true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table, err := New().ConvertToTable(context.Background(), tc.crd, &metav1.TableOptions{})
			if err != nil {
				t.Fatalf("ConvertToTable error: %v", err)
			}
			got := table.Rows[0].Cells
			if len(got) != len(tc.expect) {
				t.Fatalf("expected %d cells, got %d (%#v)", len(tc.expect), len(got), got)
			}
			for i, want := range tc.expect {
				if got[i] != want {
					t.Errorf("cell[%d] = %v, want %v", i, got[i], want)
				}
			}
		})
	}
}
