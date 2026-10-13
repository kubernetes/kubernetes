/*
Copyright 2019 The Kubernetes Authors.

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

package internal_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
	internaltesting "k8s.io/apimachinery/pkg/util/managedfields/internal/testing"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestFieldManagerUpdateNoErrors(t *testing.T) {
	fm := &fakeManager{}
	f := internaltesting.NewTestFieldManagerImpl(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"),
		"",
		func(m internal.Manager) internal.Manager {
			fm.Manager = m
			return fm
		})

	podWithLabels := func(labels ...string) runtime.Object {
		labelMap := map[string]interface{}{}
		for _, key := range labels {
			labelMap[key] = "true"
		}
		obj := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": labelMap,
				},
			},
		}
		obj.SetKind("Pod")
		obj.SetAPIVersion("v1")
		return obj
	}

	f.UpdateNoErrors(podWithLabels("one"), "fieldmanager_test_update_1")
	if len(f.ManagedFields()) == 0 {
		t.Fatalf("expected managedFields to be set, but they are empty")
	}

	before := []metav1.ManagedFieldsEntry{}
	for _, m := range f.ManagedFields() {
		before = append(before, *m.DeepCopy())
	}

	// Inject an error so UpdateNoErrors will hit the error code path.
	fm.Error = errors.New("test error")
	f.UpdateNoErrors(podWithLabels("one", "two"), "fieldmanager_test_update_1")

	if after := f.ManagedFields(); !apiequality.Semantic.DeepEqual(before, after) {
		t.Fatalf("expected idempotence, but managedFields changed:\nbefore: %v\n after: %v", mustMarshal(before), mustMarshal(after))
	}
}

var fakeTypeConverter = func() internal.TypeConverter {
	data, err := os.ReadFile(filepath.Join(
		strings.Repeat(".."+string(filepath.Separator), 8),
		"api", "openapi-spec", "swagger.json"))
	if err != nil {
		panic(err)
	}
	convertedDefs := map[string]*spec.Schema{}
	spec := spec.Swagger{}
	if err := json.Unmarshal(data, &spec); err != nil {
		panic(err)
	}

	for k, v := range spec.Definitions {
		vCopy := v
		convertedDefs[k] = &vCopy
	}

	typeConverter, err := internal.NewTypeConverter(convertedDefs, false)
	if err != nil {
		panic(err)
	}
	return typeConverter
}()
