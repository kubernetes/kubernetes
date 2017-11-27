/*
Copyright 2017 The Kubernetes Authors.

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
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	arbitraryTime = metav1.Date(2017, 11, 1, 14, 28, 47, 0, time.FixedZone("CET", 0))
)

func TestIsPVCBeingDeleted(t *testing.T) {
	tests := []struct {
		pvc  *v1.PersistentVolumeClaim
		want bool
	}{
		{
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: nil,
				},
			},
			want: false,
		},
		{
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &arbitraryTime,
				},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		if got := IsPVCBeingDeleted(tt.pvc); got != tt.want {
			t.Errorf("IsPVCBeingDeleted(%v) = %v WANT %v", tt.pvc, got, tt.want)
		}
	}
}

func TestAddProtectionFinalizer(t *testing.T) {
	tests := []struct {
		name string
		pvc  *v1.PersistentVolumeClaim
		want *v1.PersistentVolumeClaim
	}{
		{
			"PVC without finalizer",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pvc",
					Namespace: "ns",
				},
			},
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{PVCProtectionFinalizer},
				},
			},
		},
		{
			"PVC with some finalizers",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{"1", "2", "3", PVCProtectionFinalizer + "suffix", "prefix" + PVCProtectionFinalizer},
				},
			},
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{"1", "2", "3", PVCProtectionFinalizer + "suffix", "prefix" + PVCProtectionFinalizer, PVCProtectionFinalizer},
				},
			},
		},
	}
	for _, test := range tests {
		got := test.pvc.DeepCopy()
		AddProtectionFinalizer(got)
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("Test %q: expected:\n%s\n\ngot:\n%s", test.name, spew.Sdump(test.want), spew.Sdump(got))
		}
	}
}

func TestRemoveProtectionFinalizer(t *testing.T) {
	tests := []struct {
		name string
		pvc  *v1.PersistentVolumeClaim
		want *v1.PersistentVolumeClaim
	}{
		{
			"PVC without finalizer",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pvc",
					Namespace: "ns",
				},
			},
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pvc",
					Namespace: "ns",
				},
			},
		},
		{
			"PVC with finalizer",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{PVCProtectionFinalizer},
				},
			},
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pvc",
					Namespace: "ns",
				},
			},
		},
		{
			"PVC with many finalizers",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{"1", "2", "3", PVCProtectionFinalizer + "suffix", "prefix" + PVCProtectionFinalizer, PVCProtectionFinalizer},
				},
			},
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{"1", "2", "3", PVCProtectionFinalizer + "suffix", "prefix" + PVCProtectionFinalizer},
				},
			},
		},
	}
	for _, test := range tests {
		got := test.pvc.DeepCopy()
		RemoveProtectionFinalizer(got)
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("Test %q: expected:\n%s\n\ngot:\n%s", test.name, spew.Sdump(test.want), spew.Sdump(got))
		}
	}
}

func TestIsProtectionFinalizerPresent(t *testing.T) {
	tests := []struct {
		name string
		pvc  *v1.PersistentVolumeClaim
		want bool
	}{
		{
			"PVC without finalizer",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pvc",
					Namespace: "ns",
				},
			},
			false,
		},
		{
			"PVC with many unrelated finalizers",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{"1", "2", "3", PVCProtectionFinalizer + "suffix", "prefix" + PVCProtectionFinalizer},
				},
			},
			false,
		},
		{
			"PVC with many finalizers",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{"1", "2", "3", PVCProtectionFinalizer + "suffix", "prefix" + PVCProtectionFinalizer, PVCProtectionFinalizer},
				},
			},
			true,
		},
		{
			"PVC with finalizer",
			&v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pvc",
					Namespace:  "ns",
					Finalizers: []string{PVCProtectionFinalizer},
				},
			},
			true,
		},
	}
	for _, test := range tests {
		got := IsProtectionFinalizerPresent(test.pvc)
		if got != test.want {
			t.Errorf("Test %q: expected %v, got %v", test.name, test.want, got)
		}
	}
}
