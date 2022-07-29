/*
Copyright 2022 The Kubernetes Authors.

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

package storage

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsDefaultAnnotationText(t *testing.T) {
	tests := []struct {
		name         string
		obj          metav1.ObjectMeta
		expectResult string
	}{
		{
			name: "The annotation is not set",
			obj: metav1.ObjectMeta{
				Annotations: map[string]string{},
			},
			expectResult: "No",
		},
		{
			name: "The annotation is set",
			obj: metav1.ObjectMeta{
				Annotations: map[string]string{IsDefaultStorageClassAnnotation: "true"},
			},
			expectResult: "Yes",
		},
		{
			name: "The annotation is set",
			obj: metav1.ObjectMeta{
				Annotations: map[string]string{BetaIsDefaultStorageClassAnnotation: "true"},
			},
			expectResult: "Yes",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsDefaultAnnotationText(tt.obj)
			if got != tt.expectResult {
				t.Errorf("expected result %v; got %v", tt.expectResult, got)
			}
		})
	}
}

func TestGetAccessModesAsString(t *testing.T) {
	tests := []struct {
		name         string
		modes        []v1.PersistentVolumeAccessMode
		expectResult string
	}{
		{
			name: "Contains access mode",
			modes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
				v1.ReadOnlyMany,
				v1.ReadWriteMany,
				v1.ReadWriteOncePod,
			},
			expectResult: "RWO,ROX,RWX,RWOP",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetAccessModesAsString(tt.modes)
			if got != tt.expectResult {
				t.Errorf("expected result %v; got %v", tt.expectResult, got)
			}
		})
	}
}

func TestGetPersistentVolumeClass(t *testing.T) {
	tests := []struct {
		name         string
		volume       *v1.PersistentVolume
		expectResult string
	}{
		{
			name: "Get beta storage class annotation",
			volume: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{v1.BetaStorageClassAnnotation: "class"},
				},
			},
			expectResult: "class",
		},
		{
			name: "Get storage class name",
			volume: &v1.PersistentVolume{
				Spec: v1.PersistentVolumeSpec{
					StorageClassName: "class1",
				},
			},
			expectResult: "class1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetPersistentVolumeClass(tt.volume)
			if got != tt.expectResult {
				t.Errorf("expected result %v; got %v", tt.expectResult, got)
			}
		})
	}
}

func TestGetPersistentVolumeClaimClass(t *testing.T) {
	StorageClassName := "class1"
	tests := []struct {
		name         string
		claim        *v1.PersistentVolumeClaim
		expectResult string
	}{
		{
			name: "Get beta storage class annotation",
			claim: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{v1.BetaStorageClassAnnotation: "class"},
				},
			},
			expectResult: "class",
		},
		{
			name: "Get storage class name",
			claim: &v1.PersistentVolumeClaim{
				Spec: v1.PersistentVolumeClaimSpec{
					StorageClassName: &StorageClassName,
				},
			},
			expectResult: StorageClassName,
		},
		{
			name:         "No storage class",
			claim:        &v1.PersistentVolumeClaim{},
			expectResult: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetPersistentVolumeClaimClass(tt.claim)
			if got != tt.expectResult {
				t.Errorf("expected result %v; got %v", tt.expectResult, got)
			}
		})
	}
}
