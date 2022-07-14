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
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
)

var ignoreTimestampEqualities = func() conversion.Equalities {
	var eqs = equality.Semantic.Copy()

	err := eqs.AddFunc(
		func(a, b metav1.ManagedFieldsEntry) bool {
			// Two objects' managed fields are equivalent if, ignoring timestamp,
			//	the objects are deeply equal.
			a.Time = nil
			b.Time = nil
			return reflect.DeepEqual(a, b)
		},
	)

	if err != nil {
		panic(err)
	}

	return eqs
}()

// IgnoreManagedFieldsTimestampsTransformer reverts timestamp updates
// if the non-managed parts of the object are equivalent
func IgnoreManagedFieldsTimestampsTransformer(
	_ context.Context,
	newObj runtime.Object,
	oldObj runtime.Object,
) (runtime.Object, error) {
	// If managedFields modulo timestamps are unchanged
	//		and
	//	rest of object is unchanged
	//		then
	//	revert any changes to timestamps in managed fields
	//		(to prevent spurious ResourceVersion bump)
	//
	// Procecure:
	// Do a quicker check to see if just managed fields modulo timestamps are
	//	unchanged. If so, then do the full, slower check.
	//
	// In most cases which actually update the object, the managed fields modulo
	//	timestamp check will fail, and we will be able to return early.
	//
	// In other cases, the managed fields may be exactly the same,
	// 	except for timestamp, but the objects are the different. This is the
	//	slow path which checks the full object.
	oldAccessor, err := meta.Accessor(oldObj)
	if err != nil {
		return nil, fmt.Errorf("failed to acquire accessor for oldObj: %v", err)
	}

	accessor, err := meta.Accessor(newObj)
	if err != nil {
		return nil, fmt.Errorf("failed to acquire accessor for newObj: %v", err)
	}

	oldManagedFields := oldAccessor.GetManagedFields()
	newManagedFields := accessor.GetManagedFields()

	// This condition ensures the managed fields are always compared first. If
	//	this check fails, the if statement will short circuit. If the check
	// 	succeeds the slow path is taken which compares entire objects.
	if ignoreTimestampEqualities.DeepEqualWithNilDifferentFromEmpty(oldManagedFields, newManagedFields) &&
		ignoreTimestampEqualities.DeepEqualWithNilDifferentFromEmpty(newObj, oldObj) {

		// Remove any changed timestamps, so that timestamp is not the only
		// change seen by etcd.
		//
		// newManagedFields is known to be exactly pairwise equal to
		// oldManagedFields except for timestamps.
		//
		// Simply replace possibly changed new timestamps with their old values.
		for idx := 0; idx < len(oldManagedFields); idx++ {
			newManagedFields[idx].Time = oldManagedFields[idx].Time
		}

		accessor.SetManagedFields(newManagedFields)
		return newObj, nil
	}
	return newObj, nil
}
