/*
Copyright 2016 The Kubernetes Authors.

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

package garbagecollector

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
)

func TestDeleteOwnerRefPatch(t *testing.T) {
	original := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1"},
				{UID: "2"},
				{UID: "3"},
			},
		},
	}
	originalData := serilizeOrDie(t, original)
	expected := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1"},
			},
		},
	}
	patch := deleteOwnerRefPatch("100", "2", "3")
	patched, err := strategicpatch.StrategicMergePatch(originalData, patch, v1.Pod{})
	if err != nil {
		t.Fatal(err)
	}
	var got v1.Pod
	if err := json.Unmarshal(patched, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("expected: %#v,\ngot: %#v", expected, got)
	}
}

func TestUnblockOwnerReference(t *testing.T) {
	trueVar := true
	falseVar := false
	original := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1", BlockOwnerDeletion: &trueVar},
				{UID: "2", BlockOwnerDeletion: &falseVar},
				{UID: "3"},
			},
		},
	}
	originalData := serilizeOrDie(t, original)
	expected := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1", BlockOwnerDeletion: &falseVar},
				{UID: "2", BlockOwnerDeletion: &falseVar},
				{UID: "3"},
			},
		},
	}
	accessor, err := meta.Accessor(&original)
	if err != nil {
		t.Fatal(err)
	}
	n := node{
		owners: accessor.GetOwnerReferences(),
	}
	patch, err := n.patchToUnblockOwnerReferences()
	if err != nil {
		t.Fatal(err)
	}
	patched, err := strategicpatch.StrategicMergePatch(originalData, patch, v1.Pod{})
	if err != nil {
		t.Fatal(err)
	}
	var got v1.Pod
	if err := json.Unmarshal(patched, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("expected: %#v,\ngot: %#v", expected, got)
		t.Errorf("expected: %#v,\ngot: %#v", expected.OwnerReferences, got.OwnerReferences)
		for _, ref := range got.OwnerReferences {
			t.Errorf("ref.UID=%s, ref.BlockOwnerDeletion=%v", ref.UID, *ref.BlockOwnerDeletion)
		}
	}
}

func TestDeleteFinalizerPatch(t *testing.T) {
	original := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			Finalizers: []string{
				"1",
				"2",
				"3",
			},
		},
	}
	originalData := serilizeOrDie(t, original)
	expected := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			Finalizers: []string{
				"1",
				"3",
			},
		},
	}
	patch := deleteFinalizerPatch("2", "100")
	patched, err := strategicpatch.StrategicMergePatch(originalData, patch, v1.Pod{})
	if err != nil {
		t.Fatal(err)
	}
	var got v1.Pod
	if err := json.Unmarshal(patched, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("expected: %#v,\ngot: %#v", expected, got)
	}
}
