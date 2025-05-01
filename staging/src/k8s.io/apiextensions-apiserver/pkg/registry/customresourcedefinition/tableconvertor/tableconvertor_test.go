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

func TestConvertToTable_Single(t *testing.T) {
	crd := &apiextensions.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "tests.mygroup.io",
			CreationTimestamp: metav1.NewTime(time.Date(2025, 5, 3, 12, 0, 0, 0, time.UTC)),
		},
		Spec: apiextensions.CustomResourceDefinitionSpec{
			Group: "mygroup.io",
			Scope: apiextensions.ClusterScoped,
			Versions: []apiextensions.CustomResourceDefinitionVersion{
				{Name: "v1"},
				{Name: "v2"},
			},
		},
	}

	table, err := New().ConvertToTable(context.Background(), crd, &metav1.TableOptions{})
	if err != nil {
		t.Fatalf("ConvertToTable returned error: %v", err)
	}

	// Expect one row
	if len(table.Rows) != 1 {
		t.Fatalf("Expected 1 row, got %d", len(table.Rows))
	}
	cells := table.Rows[0].Cells
	// Verify cells order and content
	expected := []string{"tests.mygroup.io", "mygroup.io", string(apiextensions.ClusterScoped), "v1,v2", "2025-05-03T12:00:00Z"}
	for i, exp := range expected {
		if cells[i] != exp {
			t.Errorf("Cell %d: expected %v, got %v", i, exp, cells[i])
		}
	}
}

func TestConvertToTable_List(t *testing.T) {
	crd1 := &apiextensions.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "one.group.io", CreationTimestamp: metav1.NewTime(time.Now())},
		Spec:       apiextensions.CustomResourceDefinitionSpec{Group: "group.io", Scope: apiextensions.NamespaceScoped, Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "v1"}}},
	}
	crd2 := &apiextensions.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "two.group.io", CreationTimestamp: metav1.NewTime(time.Now())},
		Spec:       apiextensions.CustomResourceDefinitionSpec{Group: "group.io", Scope: apiextensions.NamespaceScoped, Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "v2"}}},
	}
	list := &apiextensions.CustomResourceDefinitionList{Items: []apiextensions.CustomResourceDefinition{*crd2, *crd1}}

	table, err := New().ConvertToTable(context.Background(), list, &metav1.TableOptions{})
	if err != nil {
		t.Fatalf("ConvertToTable(list) returned error: %v", err)
	}

	if len(table.Rows) != 2 {
		t.Fatalf("Expected 2 rows, got %d", len(table.Rows))
	}

	// Ensure sorting of list items preserved insertion order of list (not table)
	names := []string{table.Rows[0].Cells[0].(string), table.Rows[1].Cells[0].(string)}
	expectedNames := []string{"two.group.io", "one.group.io"}
	for i, exp := range expectedNames {
		if names[i] != exp {
			t.Errorf("Row %d Name: expected %s, got %s", i, exp, names[i])
		}
	}
}
