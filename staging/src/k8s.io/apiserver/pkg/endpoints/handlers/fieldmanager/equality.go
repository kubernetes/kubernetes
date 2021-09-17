/*
Copyright 2021 The Kubernetes Authors.

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

package fieldmanager

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
)

var equalities = func() conversion.Equalities {
	e := equality.Semantic
	e.AddFunc(ignoreManagedFieldsTimestamp)
	return e
}()

// EqualObjects returns true if the two objects are "equals".
//
// The two objects are equal if they have the exact same values, and
// their managed fieldsets are the same, except for the timestamp.
func EqualObjects(new, old runtime.Object) bool {
	return equalities.DeepEqual(new, old)
}

// DropEqualTransformer is a transformer that cancels the change if the
// objects are equal.
func DropEqualTransformer(_ context.Context, newObj, oldObj runtime.Object) (runtime.Object, error) {
	if EqualObjects(newObj, oldObj) {
		return oldObj, nil
	}
	return newObj, nil
}

func ignoreManagedFieldsTimestamp(a, b metav1.ManagedFieldsEntry) bool {
	a.Time = nil
	b.Time = nil

	return reflect.DeepEqual(a, b)
}
