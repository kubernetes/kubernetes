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

package protectionutil

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume/util"
)

const (
	defaultNS      = "default"
	defaultPVCName = "pvc1"
	defaultPVName  = "pv1"
)

type TestCase struct {
	name      string
	obj       metav1.Object
	finalizer string
	result    bool
}

func pvc() *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPVCName,
			Namespace: defaultNS,
		},
	}
}

func pv() *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultPVName,
		},
	}
}

func TestIsDeletionCandidateLackDeleteTimeAndFinalizer(t *testing.T) {

	tests := []TestCase{
		{
			name:      "pv lacks delete time and finalizer",
			obj:       pv(),
			finalizer: util.PVProtectionFinalizer,
			result:    false,
		},
		{
			name:      "pvc lacks delete time and finalizer",
			obj:       pvc(),
			finalizer: util.PVCProtectionFinalizer,

			result: false,
		},
	}
	for _, test := range tests {
		if test.result != IsDeletionCandidate(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}

func TestIsDeletionCandidateLackDeleteTime(t *testing.T) {
	pv := pv()
	pv.SetFinalizers([]string{util.PVProtectionFinalizer})
	pvc := pvc()
	pvc.SetFinalizers([]string{util.PVCProtectionFinalizer})
	tests := []TestCase{
		{
			name:      "pv lacks delete time",
			obj:       pv,
			finalizer: util.PVProtectionFinalizer,
			result:    false,
		},
		{
			name:      "pvc lacks delete time",
			obj:       pvc,
			finalizer: util.PVCProtectionFinalizer,

			result: false,
		},
	}
	for _, test := range tests {
		if test.result != IsDeletionCandidate(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}

func TestIsDeletionCandidateLackFinalizer(t *testing.T) {

	pv := pv()
	pv.SetDeletionTimestamp(&metav1.Time{})
	pvc := pvc()
	pvc.SetDeletionTimestamp(&metav1.Time{})
	tests := []TestCase{
		{
			name:      "pv lacks finalizer",
			obj:       pv,
			finalizer: util.PVProtectionFinalizer,
			result:    false,
		},
		{
			name:      "pvc lacks finalizer",
			obj:       pvc,
			finalizer: util.PVCProtectionFinalizer,
			result:    false,
		},
	}
	for _, test := range tests {
		if test.result != IsDeletionCandidate(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}

func TestIsDeletionCandidateSuccess(t *testing.T) {

	pv := pv()
	pv.SetDeletionTimestamp(&metav1.Time{})
	pv.SetFinalizers([]string{util.PVProtectionFinalizer})
	pvc := pvc()
	pvc.SetDeletionTimestamp(&metav1.Time{})
	pvc.SetFinalizers([]string{util.PVCProtectionFinalizer})

	tests := []TestCase{
		{
			name:      "pv is to delete",
			obj:       pv,
			finalizer: util.PVProtectionFinalizer,
			result:    true,
		},
		{
			name:      "pvc is to delete",
			obj:       pvc,
			finalizer: util.PVCProtectionFinalizer,
			result:    true,
		},
	}
	for _, test := range tests {
		if test.result != IsDeletionCandidate(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}

func TestNeedToAddFinalizerHasDeleteTimeAndFinalizer(t *testing.T) {
	pv := pv()
	pv.SetDeletionTimestamp(&metav1.Time{})
	pv.SetFinalizers([]string{util.PVProtectionFinalizer})
	pvc := pvc()
	pvc.SetDeletionTimestamp(&metav1.Time{})
	pvc.SetFinalizers([]string{util.PVCProtectionFinalizer})

	tests := []TestCase{
		{
			name:      "pv has delete time and finalizer",
			obj:       pv,
			finalizer: util.PVProtectionFinalizer,
			result:    false,
		},
		{
			name:      "pvc has delete time and finalizer",
			obj:       pvc,
			finalizer: util.PVCProtectionFinalizer,
			result:    false,
		},
	}
	for _, test := range tests {
		if test.result != NeedToAddFinalizer(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}

func TestNeedToAddFinalizerHasDeleteTime(t *testing.T) {
	pv := pv()
	pv.SetDeletionTimestamp(&metav1.Time{})
	pvc := pvc()
	pvc.SetDeletionTimestamp(&metav1.Time{})
	tests := []TestCase{
		{
			name:      "pv has delete",
			obj:       pv,
			finalizer: util.PVProtectionFinalizer,
			result:    false,
		},
		{
			name:      "pvc has delete",
			obj:       pvc,
			finalizer: util.PVCProtectionFinalizer,
			result:    false,
		},
	}
	for _, test := range tests {
		if test.result != NeedToAddFinalizer(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}

func TestNeedToAddFinalizerHasFinalizer(t *testing.T) {
	pv := pv()
	pv.SetFinalizers([]string{util.PVProtectionFinalizer})
	pvc := pvc()
	pvc.SetFinalizers([]string{util.PVCProtectionFinalizer})

	tests := []TestCase{
		{
			name:      "pv has finalizer",
			obj:       pv,
			finalizer: util.PVProtectionFinalizer,
			result:    false,
		},
		{
			name:      "pvc has finalizer",
			obj:       pvc,
			finalizer: util.PVCProtectionFinalizer,
			result:    false,
		},
	}
	for _, test := range tests {
		if test.result != NeedToAddFinalizer(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}

func TestNeedToAddFinalizerSuccess(t *testing.T) {
	tests := []TestCase{
		{
			name:      "pv needs add finalizer",
			obj:       pv(),
			finalizer: util.PVProtectionFinalizer,
			result:    true,
		},
		{
			name:      "pvc needs add finalizer",
			obj:       pvc(),
			finalizer: util.PVCProtectionFinalizer,
			result:    true,
		},
	}
	for _, test := range tests {
		if test.result != NeedToAddFinalizer(test.obj, test.finalizer) {
			t.Error(test.name)
		}
	}
}
