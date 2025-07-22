/*
Copyright 2023 The Kubernetes Authors.

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

package util

import (
	"testing"
	"time"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/controller"
)

func TestGetDefaultClass(t *testing.T) {

	var (
		t1 = time.Now()
		t2 = time.Now().Add(1 * time.Hour)

		sc1 = &storagev1.StorageClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "my-storage-class1",
				Annotations: map[string]string{
					"a": "b",
				},
			},
		}
		sc2 = &storagev1.StorageClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "my-storage-class2",
				Annotations: map[string]string{
					"a": "b",
				},
			},
		}

		sc3 = &storagev1.StorageClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "my-storage-class3",
				Annotations: map[string]string{
					IsDefaultStorageClassAnnotation: "true",
				},
				CreationTimestamp: metav1.Time{Time: t1},
			},
		}

		sc4 = &storagev1.StorageClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "my-storage-class4",
				Annotations: map[string]string{
					IsDefaultStorageClassAnnotation: "true",
				},
				CreationTimestamp: metav1.Time{Time: t2},
			},
		}

		sc5 = &storagev1.StorageClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "my-storage-class5",
				Annotations: map[string]string{
					IsDefaultStorageClassAnnotation: "true",
				},
				CreationTimestamp: metav1.Time{Time: t2},
			},
		}
	)

	testCases := []struct {
		name    string
		classes []*storagev1.StorageClass
		expect  *storagev1.StorageClass
	}{

		{
			name: "no storage class",
		},

		{
			name:    "no default storage class",
			classes: []*storagev1.StorageClass{sc1, sc2},
			expect:  nil,
		},

		{
			name:    "one default storage class",
			classes: []*storagev1.StorageClass{sc1, sc2, sc3},
			expect:  sc3,
		},

		{
			name:    "two default storage class with different creation timestamp",
			classes: []*storagev1.StorageClass{sc3, sc4},
			expect:  sc4,
		},

		{
			name:    "two default storage class with same creation timestamp",
			classes: []*storagev1.StorageClass{sc4, sc5},
			expect:  sc4,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
			for _, c := range tc.classes {
				informerFactory.Storage().V1().StorageClasses().Informer().GetStore().Add(c)
			}
			lister := informerFactory.Storage().V1().StorageClasses().Lister()
			actual, err := GetDefaultClass(lister)
			if err != nil {
				t.Errorf("Expected no error, got %v", err)
				return
			}
			if tc.expect != actual {
				t.Errorf("Expected %v, got %v", tc.expect, actual)
			}
		})
	}

}
