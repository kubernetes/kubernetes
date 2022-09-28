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
	"os"
	"reflect"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/klog/v2"
)

func determineAvoidNoopTimestampUpdatesEnabled() bool {
	if avoidNoopTimestampUpdatesString, exists := os.LookupEnv("KUBE_APISERVER_AVOID_NOOP_SSA_TIMESTAMP_UPDATES"); exists {
		if ret, err := strconv.ParseBool(avoidNoopTimestampUpdatesString); err == nil {
			return ret
		} else {
			klog.Errorf("failed to parse envar KUBE_APISERVER_AVOID_NOOP_SSA_TIMESTAMP_UPDATES: %v", err)
		}
	}

	// enabled by default
	return true
}

var (
	avoidNoopTimestampUpdatesEnabled = determineAvoidNoopTimestampUpdatesEnabled()
)

var avoidTimestampEqualities = func() conversion.Equalities {
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
) (res runtime.Object, err error) {
	if !avoidNoopTimestampUpdatesEnabled {
		return newObj, nil
	}

	outcome := "unequal_objects_fast"
	start := time.Now()
	err = nil
	res = nil

	defer func() {
		if err != nil {
			outcome = "error"
		}

		metrics.RecordTimestampComparisonLatency(outcome, time.Since(start))
	}()

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

	if len(oldManagedFields) != len(newManagedFields) {
		// Return early if any managed fields entry was added/removed.
		// We want to retain user expectation that even if they write to a field
		// whose value did not change, they will still result as the field
		// manager at the end.
		return newObj, nil
	} else if len(newManagedFields) == 0 {
		// This transformation only makes sense when managedFields are
		// non-empty
		return newObj, nil
	}

	// This transformation only makes sense if the managed fields has at least one
	// changed timestamp; and are otherwise equal. Return early if there are no
	// changed timestamps.
	allTimesUnchanged := true
	for i, e := range newManagedFields {
		if !e.Time.Equal(oldManagedFields[i].Time) {
			allTimesUnchanged = false
			break
		}
	}

	if allTimesUnchanged {
		return newObj, nil
	}

	// This condition ensures the managed fields are always compared first. If
	//	this check fails, the if statement will short circuit. If the check
	// 	succeeds the slow path is taken which compares entire objects.
	if !avoidTimestampEqualities.DeepEqualWithNilDifferentFromEmpty(oldManagedFields, newManagedFields) {
		return newObj, nil
	}

	if avoidTimestampEqualities.DeepEqualWithNilDifferentFromEmpty(newObj, oldObj) {
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
		outcome = "equal_objects"
		return newObj, nil
	}

	outcome = "unequal_objects_slow"
	return newObj, nil
}
