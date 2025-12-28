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

package storageclass

import (
	"context"
	"testing"
	"time"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
)

// mockLister implements Lister interface for testing
type mockLister struct {
	storageClasses []storageapi.StorageClass
}

func (m *mockLister) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	return &storageapi.StorageClassList{Items: m.storageClasses}, nil
}

func TestTableConvertor_EffectiveDefault(t *testing.T) {
	now := time.Now()
	earlier := now.Add(-1 * time.Hour)

	tests := []struct {
		name                string
		storageClasses      []storageapi.StorageClass
		expectedDefault     string
		expectedNonDefaults []string
	}{
		{
			name: "single default",
			storageClasses: []storageapi.StorageClass{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "standard",
						CreationTimestamp: metav1.Time{Time: now},
						Annotations: map[string]string{
							"storageclass.kubernetes.io/is-default-class": "true",
						},
					},
					Provisioner: "kubernetes.io/gce-pd",
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "fast",
						CreationTimestamp: metav1.Time{Time: earlier},
					},
					Provisioner: "kubernetes.io/gce-pd",
				},
			},
			expectedDefault:     "standard",
			expectedNonDefaults: []string{"fast"},
		},
		{
			name: "multiple defaults - newest wins",
			storageClasses: []storageapi.StorageClass{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "standard",
						CreationTimestamp: metav1.Time{Time: earlier},
						Annotations: map[string]string{
							"storageclass.kubernetes.io/is-default-class": "true",
						},
					},
					Provisioner: "kubernetes.io/gce-pd",
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "fast",
						CreationTimestamp: metav1.Time{Time: now},
						Annotations: map[string]string{
							"storageclass.kubernetes.io/is-default-class": "true",
						},
					},
					Provisioner: "kubernetes.io/gce-pd",
				},
			},
			expectedDefault:     "fast",
			expectedNonDefaults: []string{"standard"},
		},
		{
			name: "multiple defaults same timestamp - alphabetically first wins",
			storageClasses: []storageapi.StorageClass{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "zeta",
						CreationTimestamp: metav1.Time{Time: now},
						Annotations: map[string]string{
							"storageclass.kubernetes.io/is-default-class": "true",
						},
					},
					Provisioner: "kubernetes.io/gce-pd",
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "alpha",
						CreationTimestamp: metav1.Time{Time: now},
						Annotations: map[string]string{
							"storageclass.kubernetes.io/is-default-class": "true",
						},
					},
					Provisioner: "kubernetes.io/gce-pd",
				},
			},
			expectedDefault:     "alpha",
			expectedNonDefaults: []string{"zeta"},
		},
		{
			name: "no defaults",
			storageClasses: []storageapi.StorageClass{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "standard",
						CreationTimestamp: metav1.Time{Time: now},
					},
					Provisioner: "kubernetes.io/gce-pd",
				},
			},
			expectedDefault:     "",
			expectedNonDefaults: []string{"standard"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lister := &mockLister{storageClasses: tt.storageClasses}
			convertor := NewTableConvertor()
			convertor.SetLister(lister)

			// Test list conversion
			list := &storageapi.StorageClassList{Items: tt.storageClasses}
			table, err := convertor.ConvertToTable(context.Background(), list, nil)
			if err != nil {
				t.Fatalf("ConvertToTable failed: %v", err)
			}

			// Verify default marking in table rows
			for _, row := range table.Rows {
				if len(row.Cells) == 0 {
					continue
				}
				name, ok := row.Cells[0].(string)
				if !ok {
					continue
				}

				hasDefaultSuffix := len(name) > len(" (default)") && name[len(name)-len(" (default)"):] == " (default)"
				cleanName := name
				if hasDefaultSuffix {
					cleanName = name[:len(name)-len(" (default)")]
				}

				if cleanName == tt.expectedDefault {
					if !hasDefaultSuffix {
						t.Errorf("expected %q to be marked as default, but it wasn't", cleanName)
					}
				} else {
					if hasDefaultSuffix {
						t.Errorf("expected %q to NOT be marked as default, but it was", cleanName)
					}
				}
			}
		})
	}
}

func TestTableConvertor_SingleObject(t *testing.T) {
	now := time.Now()
	earlier := now.Add(-1 * time.Hour)

	// Create a scenario where we have multiple defaults
	allStorageClasses := []storageapi.StorageClass{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "standard",
				CreationTimestamp: metav1.Time{Time: earlier},
				Annotations: map[string]string{
					"storageclass.kubernetes.io/is-default-class": "true",
				},
			},
			Provisioner: "kubernetes.io/gce-pd",
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "fast",
				CreationTimestamp: metav1.Time{Time: now},
				Annotations: map[string]string{
					"storageclass.kubernetes.io/is-default-class": "true",
				},
			},
			Provisioner: "kubernetes.io/gce-pd",
		},
	}

	lister := &mockLister{storageClasses: allStorageClasses}
	convertor := NewTableConvertor()
	convertor.SetLister(lister)

	tests := []struct {
		name           string
		storageClass   *storageapi.StorageClass
		expectDefault  bool
	}{
		{
			name:          "query effective default - should show default",
			storageClass:  &allStorageClasses[1], // fast - the effective default
			expectDefault: true,
		},
		{
			name:          "query non-effective default - should NOT show default",
			storageClass:  &allStorageClasses[0], // standard - has annotation but not effective
			expectDefault: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			table, err := convertor.ConvertToTable(context.Background(), tt.storageClass, nil)
			if err != nil {
				t.Fatalf("ConvertToTable failed: %v", err)
			}

			if len(table.Rows) == 0 || len(table.Rows[0].Cells) == 0 {
				t.Fatal("expected at least one row with cells")
			}

			name, ok := table.Rows[0].Cells[0].(string)
			if !ok {
				t.Fatalf("expected first cell to be string, got %T", table.Rows[0].Cells[0])
			}

			hasDefaultSuffix := len(name) > len(" (default)") && name[len(name)-len(" (default)"):] == " (default)"

			if tt.expectDefault && !hasDefaultSuffix {
				t.Errorf("expected %q to be marked as default, but got %q", tt.storageClass.Name, name)
			}
			if !tt.expectDefault && hasDefaultSuffix {
				t.Errorf("expected %q to NOT be marked as default, but got %q", tt.storageClass.Name, name)
			}
		})
	}
}