/*
Copyright 2020 The Kubernetes Authors.

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

package csistoragecapacity

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/storage"
)

// getValidCSIStorageCapacity returns a fully-populated CSIStorageCapacity.
func getValidCSIStorageCapacity(name string, capacityStr string) *storage.CSIStorageCapacity {
	mib := resource.MustParse("1Mi")
	c := &storage.CSIStorageCapacity{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "1",
		},
		StorageClassName: "bar",
		NodeTopology: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      "node",
					Operator: metav1.LabelSelectorOpIn,
					Values: []string{
						"node1",
					},
				},
			},
		},
		Capacity: &mib,
	}
	if capacityStr != "" {
		capacityQuantity := resource.MustParse(capacityStr)
		c.Capacity = &capacityQuantity
	}
	return c
}

func TestCSIStorageCapacityStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1alphav1",
		Resource:   "csistoragecapacities",
	})
	if !Strategy.NamespaceScoped() {
		t.Errorf("CSIStorageCapacity must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("CSIStorageCapacity should not allow create on update")
	}

	capacity := getValidCSIStorageCapacity("valid", "")
	original := capacity.DeepCopy()
	Strategy.PrepareForCreate(ctx, capacity)
	errs := Strategy.Validate(ctx, capacity)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	// Create with status should have kept status and all other fields.
	if !apiequality.Semantic.DeepEqual(capacity, original) {
		t.Errorf("unexpected objects difference after creation: %v", cmp.Diff(original, capacity))
	}

	// Update of immutable fields is disallowed
	fields := []struct {
		name   string
		update func(capacity *storage.CSIStorageCapacity)
	}{
		{
			name: "Topology",
			update: func(capacity *storage.CSIStorageCapacity) {
				capacity.NodeTopology.MatchLabels = map[string]string{"some-label": "some-value"}
			},
		},
		{
			name: "StorageClass",
			update: func(capacity *storage.CSIStorageCapacity) {
				capacity.StorageClassName += "-suffix"
			},
		},
	}
	for _, field := range fields {
		t.Run(field.name, func(t *testing.T) {
			newCapacity := capacity.DeepCopy()
			field.update(newCapacity)
			Strategy.PrepareForUpdate(ctx, newCapacity, capacity)
			errs = Strategy.ValidateUpdate(ctx, newCapacity, capacity)
			if len(errs) == 0 {
				t.Errorf("Expected a validation error")
			}
		})
	}
}

func TestCSIStorageCapacityValidation(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1alphav1",
		Resource:   "csistoragecapacities",
	})

	tests := []struct {
		name        string
		expectError bool
		old, update *storage.CSIStorageCapacity
	}{
		{
			name:   "before: no capacity, update: 1Gi capacity",
			old:    getValidCSIStorageCapacity("test", ""),
			update: getValidCSIStorageCapacity("test", "1Gi"),
		},
		{
			name:   "before: 1Gi capacity, update: no capacity",
			old:    getValidCSIStorageCapacity("test", "1Gi"),
			update: getValidCSIStorageCapacity("test", ""),
		},
		{
			name:        "name change",
			expectError: true,
			old:         getValidCSIStorageCapacity("a", ""),
			update:      getValidCSIStorageCapacity("b", ""),
		},
		{
			name:        "storage class name change",
			expectError: true,
			old:         getValidCSIStorageCapacity("test", ""),
			update: func() *storage.CSIStorageCapacity {
				capacity := getValidCSIStorageCapacity("test", "")
				capacity.StorageClassName += "-update"
				return capacity
			}(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			oldCapacity := test.old.DeepCopy()
			Strategy.PrepareForCreate(ctx, oldCapacity)
			errs := Strategy.Validate(ctx, oldCapacity)
			if len(errs) != 0 {
				t.Errorf("unexpected validating errors for create: %v", errs)
			}

			newCapacity := test.update.DeepCopy()
			Strategy.PrepareForUpdate(ctx, newCapacity, test.old)
			errs = Strategy.ValidateUpdate(ctx, newCapacity, oldCapacity)
			if len(errs) > 0 && !test.expectError {
				t.Errorf("unexpected validation failure: %+v", errs)
			}
			if len(errs) == 0 && test.expectError {
				t.Errorf("validation unexpectedly succeeded")
			}
		})
	}
}
