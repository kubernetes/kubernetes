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

	storagev1alpha1 "k8s.io/api/storage/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/controller"
)

func TestGetDefaultVolumeAttributesClass(t *testing.T) {
	var (
		t1 = time.Now()
		t2 = time.Now().Add(1 * time.Hour)
	)

	dirverName1 := "my-driver1"
	vac1 := &storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "my-vac1",
			Annotations: map[string]string{
				"a": "b",
			},
		},
		DriverName: dirverName1,
	}
	vac2 := &storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "my-vac2",
			Annotations: map[string]string{
				"a": "b",
			},
		},
		DriverName: dirverName1,
	}
	vac3 := &storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "my-vac3",
			Annotations: map[string]string{
				AlphaIsDefaultVolumeAttributesClassAnnotation: "true",
			},
			CreationTimestamp: metav1.Time{Time: t1},
		},
		DriverName: dirverName1,
	}
	vac4 := &storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "my-vac4",
			Annotations: map[string]string{
				AlphaIsDefaultVolumeAttributesClassAnnotation: "true",
			},
			CreationTimestamp: metav1.Time{Time: t2},
		},
		DriverName: dirverName1,
	}
	vac5 := &storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "my-vac5",
			Annotations: map[string]string{
				AlphaIsDefaultVolumeAttributesClassAnnotation: "true",
			},
			CreationTimestamp: metav1.Time{Time: t2},
		},
		DriverName: dirverName1,
	}

	dirverName2 := "my-driver2"
	vac6 := &storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "my-vac6",
			Annotations: map[string]string{
				"a": "b",
			},
		},
		DriverName: dirverName2,
	}
	vac7 := &storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "my-vac7",
			Annotations: map[string]string{
				AlphaIsDefaultVolumeAttributesClassAnnotation: "true",
			},
		},
		DriverName: dirverName2,
	}

	testCases := []struct {
		name       string
		driverName string
		classes    []*storagev1alpha1.VolumeAttributesClass
		expect     *storagev1alpha1.VolumeAttributesClass
	}{
		{
			name:       "no volume attributes class",
			driverName: dirverName1,
		},
		{
			name:       "no default volume attributes class",
			driverName: dirverName1,
			classes:    []*storagev1alpha1.VolumeAttributesClass{vac1, vac2, vac6},
			expect:     nil,
		},
		{
			name:       "no default volume attributes class for the driverName1",
			driverName: dirverName1,
			classes:    []*storagev1alpha1.VolumeAttributesClass{vac1, vac2, vac6, vac7},
			expect:     nil,
		},
		{
			name:       "one default volume attributes class for the driverName1",
			driverName: dirverName1,
			classes:    []*storagev1alpha1.VolumeAttributesClass{vac1, vac2, vac3, vac6, vac7},
			expect:     vac3,
		},
		{
			name:       "two default volume attributes class with different creation timestamp for the driverName1",
			driverName: dirverName1,
			classes:    []*storagev1alpha1.VolumeAttributesClass{vac3, vac4, vac6, vac7},
			expect:     vac4,
		},
		{
			name:       "two default volume attributes class with same creation timestamp for the driverName1",
			driverName: dirverName1,
			classes:    []*storagev1alpha1.VolumeAttributesClass{vac4, vac5, vac6, vac7},
			expect:     vac4,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
			for _, c := range tc.classes {
				err := informerFactory.Storage().V1alpha1().VolumeAttributesClasses().Informer().GetStore().Add(c)
				if err != nil {
					t.Errorf("Expected no error, got %v", err)
					return
				}
			}
			lister := informerFactory.Storage().V1alpha1().VolumeAttributesClasses().Lister()
			actual, err := GetDefaultVolumeAttributesClass(lister, tc.driverName)
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

func TestIsDefaultVolumeAttributesClassAnnotation(t *testing.T) {
	testCases := []struct {
		name   string
		class  *storagev1alpha1.VolumeAttributesClass
		expect bool
	}{
		{
			name:   "no annotation",
			class:  &storagev1alpha1.VolumeAttributesClass{},
			expect: false,
		},
		{
			name: "annotation is not boolean",
			class: &storagev1alpha1.VolumeAttributesClass{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						AlphaIsDefaultVolumeAttributesClassAnnotation: "not-boolean",
					},
				},
			},
			expect: false,
		},
		{
			name: "annotation is false",
			class: &storagev1alpha1.VolumeAttributesClass{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						AlphaIsDefaultVolumeAttributesClassAnnotation: "false",
					},
				},
			},
			expect: false,
		},
		{
			name: "annotation is true",
			class: &storagev1alpha1.VolumeAttributesClass{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						AlphaIsDefaultVolumeAttributesClassAnnotation: "true",
					},
				},
			},
			expect: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := IsDefaultVolumeAttributesClassAnnotation(tc.class.ObjectMeta)
			if tc.expect != actual {
				t.Errorf("Expected %v, got %v", tc.expect, actual)
			}
		})
	}
}
