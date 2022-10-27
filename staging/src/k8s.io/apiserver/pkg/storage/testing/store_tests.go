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

package testing

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	utilpointer "k8s.io/utils/pointer"
)

type KeyValidation func(ctx context.Context, t *testing.T, key string)

func RunTestCreate(ctx context.Context, t *testing.T, store storage.Interface, validation KeyValidation) {
	key := "/testkey"
	out := &example.Pod{}
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", SelfLink: "testlink"}}

	// verify that kv pair is empty before set
	if err := store.Get(ctx, key, storage.GetOptions{}, out); !storage.IsNotFound(err) {
		t.Fatalf("expecting empty result on key %s, got %v", key, err)
	}

	if err := store.Create(ctx, key, obj, out, 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	// basic tests of the output
	if obj.ObjectMeta.Name != out.ObjectMeta.Name {
		t.Errorf("pod name want=%s, get=%s", obj.ObjectMeta.Name, out.ObjectMeta.Name)
	}
	if out.ResourceVersion == "" {
		t.Errorf("output should have non-empty resource version")
	}
	if out.SelfLink != "" {
		t.Errorf("output should have empty selfLink")
	}

	validation(ctx, t, key)
}

func RunTestCreateWithTTL(ctx context.Context, t *testing.T, store storage.Interface) {
	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &example.Pod{}
	if err := store.Create(ctx, key, input, out, 1); err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: out.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	TestCheckEventType(t, watch.Deleted, w)
}

func RunTestCreateWithKeyExist(ctx context.Context, t *testing.T, store storage.Interface) {
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key, _ := TestPropagateStore(ctx, t, store, obj)
	out := &example.Pod{}
	err := store.Create(ctx, key, obj, out, 0)
	if err == nil || !storage.IsExist(err) {
		t.Errorf("expecting key exists error, but get: %s", err)
	}
}

func RunTestGet(ctx context.Context, t *testing.T, store storage.Interface) {
	// create an object to test
	key, createdObj := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	// update the object once to allow get by exact resource version to be tested
	updateObj := createdObj.DeepCopy()
	updateObj.Annotations = map[string]string{"test-annotation": "1"}
	storedObj := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, storedObj, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			ttl := uint64(1)
			return updateObj, &ttl, nil
		}, nil)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	// create an additional object to increment the resource version for pods above the resource version of the foo object
	lastUpdatedObj := &example.Pod{}
	if err := store.Create(ctx, "bar", &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}, lastUpdatedObj, 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}

	currentRV, _ := strconv.Atoi(storedObj.ResourceVersion)
	lastUpdatedCurrentRV, _ := strconv.Atoi(lastUpdatedObj.ResourceVersion)

	// TODO(jpbetz): Add exact test cases
	tests := []struct {
		name              string
		key               string
		ignoreNotFound    bool
		expectNotFoundErr bool
		expectRVTooLarge  bool
		expectedOut       []*example.Pod
		rv                string
	}{{
		name:              "get existing",
		key:               key,
		ignoreNotFound:    false,
		expectNotFoundErr: false,
		expectedOut:       []*example.Pod{storedObj},
	}, {
		// For RV=0 arbitrarily old version is allowed, including from the moment
		// when the object didn't yet exist.
		// As a result, we allow it by setting ignoreNotFound and allowing an empty
		// object in expectedOut.
		name:           "resource version 0",
		key:            key,
		ignoreNotFound: true,
		expectedOut:    []*example.Pod{{}, createdObj, storedObj},
		rv:             "0",
	}, {
		// Given that Get with set ResourceVersion is effectively always
		// NotOlderThan semantic, both versions of object are allowed.
		name:        "object created resource version",
		key:         key,
		expectedOut: []*example.Pod{createdObj, storedObj},
		rv:          createdObj.ResourceVersion,
	}, {
		name:        "current object resource version, match=NotOlderThan",
		key:         key,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", currentRV),
	}, {
		name:        "latest resource version",
		key:         key,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", lastUpdatedCurrentRV),
	}, {
		name:             "too high resource version",
		key:              key,
		expectRVTooLarge: true,
		rv:               strconv.FormatInt(math.MaxInt64, 10),
	}, {
		name:              "get non-existing",
		key:               "/non-existing",
		ignoreNotFound:    false,
		expectNotFoundErr: true,
	}, {
		name:              "get non-existing, ignore not found",
		key:               "/non-existing",
		ignoreNotFound:    true,
		expectNotFoundErr: false,
		expectedOut:       []*example.Pod{{}},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &example.Pod{}
			err := store.Get(ctx, tt.key, storage.GetOptions{IgnoreNotFound: tt.ignoreNotFound, ResourceVersion: tt.rv}, out)
			if tt.expectNotFoundErr {
				if err == nil || !storage.IsNotFound(err) {
					t.Errorf("expecting not found error, but get: %v", err)
				}
				return
			}
			if tt.expectRVTooLarge {
				if err == nil || !storage.IsTooLargeResourceVersion(err) {
					t.Errorf("expecting resource version too high error, but get: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Get failed: %v", err)
			}

			if len(tt.expectedOut) == 1 {
				ExpectNoDiff(t, fmt.Sprintf("%s: incorrect pod", tt.name), tt.expectedOut[0], out)
			} else {
				toInterfaceSlice := func(pods []*example.Pod) []interface{} {
					result := make([]interface{}, 0, len(pods))
					for i := range pods {
						result = append(result, pods[i])
					}
					return result
				}
				ExpectContains(t, fmt.Sprintf("%s: incorrect pod", tt.name), toInterfaceSlice(tt.expectedOut), out)
			}
		})
	}
}

func RunTestUnconditionalDelete(ctx context.Context, t *testing.T, store storage.Interface) {
	key, storedObj := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	tests := []struct {
		name              string
		key               string
		expectedObj       *example.Pod
		expectNotFoundErr bool
	}{{
		name:              "existing key",
		key:               key,
		expectedObj:       storedObj,
		expectNotFoundErr: false,
	}, {
		name:              "non-existing key",
		key:               "/non-existing",
		expectedObj:       nil,
		expectNotFoundErr: true,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &example.Pod{} // reset
			err := store.Delete(ctx, tt.key, out, nil, storage.ValidateAllObjectFunc, nil)
			if tt.expectNotFoundErr {
				if err == nil || !storage.IsNotFound(err) {
					t.Errorf("expecting not found error, but get: %s", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Delete failed: %v", err)
			}
			// We expect the resource version of the returned object to be
			// updated compared to the last existing object.
			if storedObj.ResourceVersion == out.ResourceVersion {
				t.Errorf("expecting resource version to be updated, but get: %s", out.ResourceVersion)
			}
			out.ResourceVersion = storedObj.ResourceVersion
			ExpectNoDiff(t, "incorrect pod:", tt.expectedObj, out)
		})
	}
}

func RunTestConditionalDelete(ctx context.Context, t *testing.T, store storage.Interface) {
	key, storedObj := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}})

	tests := []struct {
		name                string
		precondition        *storage.Preconditions
		expectInvalidObjErr bool
	}{{
		name:                "UID match",
		precondition:        storage.NewUIDPreconditions("A"),
		expectInvalidObjErr: false,
	}, {
		name:                "UID mismatch",
		precondition:        storage.NewUIDPreconditions("B"),
		expectInvalidObjErr: true,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &example.Pod{}
			err := store.Delete(ctx, key, out, tt.precondition, storage.ValidateAllObjectFunc, nil)
			if tt.expectInvalidObjErr {
				if err == nil || !storage.IsInvalidObj(err) {
					t.Errorf("expecting invalid UID error, but get: %s", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Delete failed: %v", err)
			}
			// We expect the resource version of the returned object to be
			// updated compared to the last existing object.
			if storedObj.ResourceVersion == out.ResourceVersion {
				t.Errorf("expecting resource version to be updated, but get: %s", out.ResourceVersion)
			}
			out.ResourceVersion = storedObj.ResourceVersion
			ExpectNoDiff(t, "incorrect pod:", storedObj, out)
			key, storedObj = TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}})
		})
	}
}

// The following set of Delete tests are testing the logic of adding `suggestion`
// as a parameter with probably value of the current state.
// Introducing it for GuaranteedUpdate cause a number of issues, so we're addressing
// all of those upfront by adding appropriate tests:
// - https://github.com/kubernetes/kubernetes/pull/35415
//   [DONE] Lack of tests originally - added TestDeleteWithSuggestion.
// - https://github.com/kubernetes/kubernetes/pull/40664
//   [DONE] Irrelevant for delete, as Delete doesn't write data (nor compare it).
// - https://github.com/kubernetes/kubernetes/pull/47703
//   [DONE] Irrelevant for delete, because Delete doesn't persist data.
// - https://github.com/kubernetes/kubernetes/pull/48394/
//   [DONE] Irrelevant for delete, because Delete doesn't compare data.
// - https://github.com/kubernetes/kubernetes/pull/43152
//   [DONE] Added TestDeleteWithSuggestionAndConflict
// - https://github.com/kubernetes/kubernetes/pull/54780
//   [DONE] Irrelevant for delete, because Delete doesn't compare data.
// - https://github.com/kubernetes/kubernetes/pull/58375
//   [DONE] Irrelevant for delete, because Delete doesn't compare data.
// - https://github.com/kubernetes/kubernetes/pull/77619
//   [DONE] Added TestValidateDeletionWithSuggestion for corresponding delete checks.
// - https://github.com/kubernetes/kubernetes/pull/78713
//   [DONE] Bug was in getState function which is shared with the new code.
// - https://github.com/kubernetes/kubernetes/pull/78713
//   [DONE] Added TestPreconditionalDeleteWithSuggestion

func RunTestDeleteWithSuggestion(ctx context.Context, t *testing.T, store storage.Interface) {
	key, originalPod := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func RunTestDeleteWithSuggestionAndConflict(ctx context.Context, t *testing.T, store storage.Interface) {
	key, originalPod := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// First update, so originalPod is outdated.
	updatedPod := &example.Pod{}
	if err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.ObjectMeta.Labels = map[string]string{"foo": "bar"}
			return pod, nil
		}), nil); err != nil {
		t.Errorf("Unexpected failure during updated: %v", err)
	}

	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func RunTestDeleteWithSuggestionOfDeletedObject(ctx context.Context, t *testing.T, store storage.Interface) {
	key, originalPod := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// First delete, so originalPod is outdated.
	deletedPod := &example.Pod{}
	if err := store.Delete(ctx, key, deletedPod, nil, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	// Now try deleting with stale object.
	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, storage.ValidateAllObjectFunc, originalPod); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error during deletion: %v, expected not-found", err)
	}
}

func RunTestValidateDeletionWithSuggestion(ctx context.Context, t *testing.T, store storage.Interface) {
	key, originalPod := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// Check that validaing fresh object fails is called once and fails.
	validationCalls := 0
	validationError := fmt.Errorf("validation error")
	validateNothing := func(_ context.Context, _ runtime.Object) error {
		validationCalls++
		return validationError
	}
	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, nil, validateNothing, originalPod); err != validationError {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}
	if validationCalls != 1 {
		t.Errorf("validate function should have been called once, called %d", validationCalls)
	}

	// First update, so originalPod is outdated.
	updatedPod := &example.Pod{}
	if err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.ObjectMeta.Labels = map[string]string{"foo": "bar"}
			return pod, nil
		}), nil); err != nil {
		t.Errorf("Unexpected failure during updated: %v", err)
	}

	calls := 0
	validateFresh := func(_ context.Context, obj runtime.Object) error {
		calls++
		pod := obj.(*example.Pod)
		if pod.ObjectMeta.Labels == nil || pod.ObjectMeta.Labels["foo"] != "bar" {
			return fmt.Errorf("stale object")
		}
		return nil
	}

	if err := store.Delete(ctx, key, out, nil, validateFresh, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if calls != 2 {
		t.Errorf("validate function should have been called twice, called %d", calls)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func RunTestPreconditionalDeleteWithSuggestion(ctx context.Context, t *testing.T, store storage.Interface) {
	key, originalPod := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "name"}})

	// First update, so originalPod is outdated.
	updatedPod := &example.Pod{}
	if err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.ObjectMeta.UID = "myUID"
			return pod, nil
		}), nil); err != nil {
		t.Errorf("Unexpected failure during updated: %v", err)
	}

	prec := storage.NewUIDPreconditions("myUID")

	out := &example.Pod{}
	if err := store.Delete(ctx, key, out, prec, storage.ValidateAllObjectFunc, originalPod); err != nil {
		t.Errorf("Unexpected failure during deletion: %v", err)
	}

	if err := store.Get(ctx, key, storage.GetOptions{}, &example.Pod{}); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error on reading object: %v", err)
	}
}

func RunTestList(ctx context.Context, t *testing.T, store storage.Interface) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RemainingItemCount, true)()

	initialRV, preset, err := seedMultiLevelData(ctx, store)
	if err != nil {
		t.Fatal(err)
	}

	list := &example.PodList{}
	storageOpts := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       storage.Everything,
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/two-level", storageOpts, list); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	continueRV, _ := strconv.Atoi(list.ResourceVersion)
	secondContinuation, err := storage.EncodeContinue("/two-level/2", "/two-level/", int64(continueRV))
	if err != nil {
		t.Fatal(err)
	}

	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod := obj.(*example.Pod)
		return nil, fields.Set{"metadata.name": pod.Name}, nil
	}

	tests := []struct {
		name                       string
		rv                         string
		rvMatch                    metav1.ResourceVersionMatch
		prefix                     string
		pred                       storage.SelectionPredicate
		expectedOut                []*example.Pod
		expectContinue             bool
		expectedRemainingItemCount *int64
		expectError                bool
		expectRVTooLarge           bool
		expectRV                   string
		expectRVFunc               func(string) error
	}{
		{
			name:        "rejects invalid resource version",
			prefix:      "/",
			pred:        storage.Everything,
			rv:          "abc",
			expectError: true,
		},
		{
			name:   "rejects resource version and continue token",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Label:    labels.Everything(),
				Field:    fields.Everything(),
				Limit:    1,
				Continue: secondContinuation,
			},
			rv:          "1",
			expectError: true,
		},
		{
			name:             "rejects resource version set too high",
			prefix:           "/",
			rv:               strconv.FormatInt(math.MaxInt64, 10),
			expectRVTooLarge: true,
		},
		{
			name:        "test List on existing key",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0]},
		},
		{
			name:        "test List on existing key with resource version set to 0",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0]},
			rv:          "0",
		},
		{
			name:        "test List on existing key with resource version set before first write, match=Exact",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{},
			rv:          initialRV,
			rvMatch:     metav1.ResourceVersionMatchExact,
			expectRV:    initialRV,
		},
		{
			name:        "test List on existing key with resource version set to 0, match=NotOlderThan",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0]},
			rv:          "0",
			rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:        "test List on existing key with resource version set to 0, match=Invalid",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			rv:          "0",
			rvMatch:     "Invalid",
			expectError: true,
		},
		{
			name:        "test List on existing key with resource version set before first write, match=NotOlderThan",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0]},
			rv:          initialRV,
			rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:        "test List on existing key with resource version set before first write, match=Invalid",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			rv:          initialRV,
			rvMatch:     "Invalid",
			expectError: true,
		},
		{
			name:        "test List on existing key with resource version set to current resource version",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0]},
			rv:          list.ResourceVersion,
		},
		{
			name:        "test List on existing key with resource version set to current resource version, match=Exact",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0]},
			rv:          list.ResourceVersion,
			rvMatch:     metav1.ResourceVersionMatchExact,
			expectRV:    list.ResourceVersion,
		},
		{
			name:        "test List on existing key with resource version set to current resource version, match=NotOlderThan",
			prefix:      "/one-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[0]},
			rv:          list.ResourceVersion,
			rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
		},
		{
			name:        "test List on non-existing key",
			prefix:      "/non-existing/",
			pred:        storage.Everything,
			expectedOut: nil,
		},
		{
			name:   "test List with pod name matching",
			prefix: "/one-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.ParseSelectorOrDie("metadata.name!=foo"),
			},
			expectedOut: nil,
		},
		{
			name:   "test List with limit",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1]},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
		},
		{
			name:   "test List with limit at current resource version",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1]},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         list.ResourceVersion,
			expectRV:                   list.ResourceVersion,
		},
		{
			name:   "test List with limit at current resource version and match=Exact",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1]},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         list.ResourceVersion,
			rvMatch:                    metav1.ResourceVersionMatchExact,
			expectRV:                   list.ResourceVersion,
		},
		{
			name:   "test List with limit at resource version 0",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1]},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         "0",
			expectRVFunc:               ResourceVersionNotOlderThan(list.ResourceVersion),
		},
		{
			name:   "test List with limit at resource version 0 match=NotOlderThan",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:                []*example.Pod{preset[1]},
			expectContinue:             true,
			expectedRemainingItemCount: utilpointer.Int64Ptr(1),
			rv:                         "0",
			rvMatch:                    metav1.ResourceVersionMatchNotOlderThan,
			expectRVFunc:               ResourceVersionNotOlderThan(list.ResourceVersion),
		},
		{
			name:   "test List with limit at resource version before first write and match=Exact",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{},
			expectContinue: false,
			rv:             initialRV,
			rvMatch:        metav1.ResourceVersionMatchExact,
			expectRV:       initialRV,
		},
		{
			name:   "test List with pregenerated continue token",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label:    labels.Everything(),
				Field:    fields.Everything(),
				Limit:    1,
				Continue: secondContinuation,
			},
			expectedOut: []*example.Pod{preset[2]},
		},
		{
			name:   "ignores resource version 0 for List with pregenerated continue token",
			prefix: "/two-level/",
			pred: storage.SelectionPredicate{
				Label:    labels.Everything(),
				Field:    fields.Everything(),
				Limit:    1,
				Continue: secondContinuation,
			},
			rv:          "0",
			expectedOut: []*example.Pod{preset[2]},
		},
		{
			name:        "test List with multiple levels of directories and expect flattened result",
			prefix:      "/two-level/",
			pred:        storage.Everything,
			expectedOut: []*example.Pod{preset[1], preset[2]},
		},
		{
			name:   "test List with filter returning only one item, ensure only a single page returned",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{preset[3]},
			expectContinue: true,
		},
		{
			name:   "test List with filter returning only one item, covers the entire list",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 2,
			},
			expectedOut:    []*example.Pod{preset[3]},
			expectContinue: false,
		},
		{
			name:   "test List with filter returning only one item, covers the entire list, with resource version 0",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 2,
			},
			rv:             "0",
			expectedOut:    []*example.Pod{preset[3]},
			expectContinue: false,
		},
		{
			name:   "test List with filter returning two items, more pages possible",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "foo"),
				Label: labels.Everything(),
				Limit: 2,
			},
			expectContinue: true,
			expectedOut:    []*example.Pod{preset[0], preset[1]},
		},
		{
			name:   "filter returns two items split across multiple pages",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "bar"),
				Label: labels.Everything(),
				Limit: 2,
			},
			expectedOut: []*example.Pod{preset[2], preset[4]},
		},
		{
			name:   "filter returns one item for last page, ends on last item, not full",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field:    fields.OneTermEqualSelector("metadata.name", "bar"),
				Label:    labels.Everything(),
				Limit:    2,
				Continue: EncodeContinueOrDie("z-level/3", int64(continueRV)),
			},
			expectedOut: []*example.Pod{preset[4]},
		},
		{
			name:   "filter returns one item for last page, starts on last item, full",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field:    fields.OneTermEqualSelector("metadata.name", "bar"),
				Label:    labels.Everything(),
				Limit:    1,
				Continue: EncodeContinueOrDie("z-level/3/test-2", int64(continueRV)),
			},
			expectedOut: []*example.Pod{preset[4]},
		},
		{
			name:   "filter returns one item for last page, starts on last item, partial page",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field:    fields.OneTermEqualSelector("metadata.name", "bar"),
				Label:    labels.Everything(),
				Limit:    2,
				Continue: EncodeContinueOrDie("z-level/3/test-2", int64(continueRV)),
			},
			expectedOut: []*example.Pod{preset[4]},
		},
		{
			name:   "filter returns two items, page size equal to total list size",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "bar"),
				Label: labels.Everything(),
				Limit: 5,
			},
			expectedOut: []*example.Pod{preset[2], preset[4]},
		},
		{
			name:   "filter returns one item, page size equal to total list size",
			prefix: "/",
			pred: storage.SelectionPredicate{
				Field: fields.OneTermEqualSelector("metadata.name", "fourth"),
				Label: labels.Everything(),
				Limit: 5,
			},
			expectedOut: []*example.Pod{preset[3]},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.pred.GetAttrs == nil {
				tt.pred.GetAttrs = getAttrs
			}

			out := &example.PodList{}
			storageOpts := storage.ListOptions{
				ResourceVersion:      tt.rv,
				ResourceVersionMatch: tt.rvMatch,
				Predicate:            tt.pred,
				Recursive:            true,
			}
			err = store.GetList(ctx, tt.prefix, storageOpts, out)
			if tt.expectRVTooLarge {
				if err == nil || !storage.IsTooLargeResourceVersion(err) {
					t.Fatalf("expecting resource version too high error, but get: %s", err)
				}
				return
			}

			if err != nil {
				if !tt.expectError {
					t.Fatalf("GetList failed: %v", err)
				}
				return
			}
			if tt.expectError {
				t.Fatalf("expected error but got none")
			}
			if (len(out.Continue) > 0) != tt.expectContinue {
				t.Errorf("unexpected continue token: %q", out.Continue)
			}

			// If a client requests an exact resource version, it must be echoed back to them.
			if tt.expectRV != "" {
				if tt.expectRV != out.ResourceVersion {
					t.Errorf("resourceVersion in list response want=%s, got=%s", tt.expectRV, out.ResourceVersion)
				}
			}
			if tt.expectRVFunc != nil {
				if err := tt.expectRVFunc(out.ResourceVersion); err != nil {
					t.Errorf("resourceVersion in list response invalid: %v", err)
				}
			}
			if len(tt.expectedOut) != len(out.Items) {
				t.Fatalf("length of list want=%d, got=%d", len(tt.expectedOut), len(out.Items))
			}
			if diff := cmp.Diff(tt.expectedRemainingItemCount, out.ListMeta.GetRemainingItemCount()); diff != "" {
				t.Errorf("incorrect remainingItemCount: %s", diff)
			}
			for j, wantPod := range tt.expectedOut {
				getPod := &out.Items[j]
				ExpectNoDiff(t, fmt.Sprintf("%s: incorrect pod", tt.name), wantPod, getPod)
			}
		})
	}
}

func RunTestListWithoutPaging(ctx context.Context, t *testing.T, store storage.Interface) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RemainingItemCount, true)()

	_, preset, err := seedMultiLevelData(ctx, store)
	if err != nil {
		t.Fatal(err)
	}

	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod := obj.(*example.Pod)
		return nil, fields.Set{"metadata.name": pod.Name}, nil
	}

	tests := []struct {
		name                       string
		disablePaging              bool
		rv                         string
		rvMatch                    metav1.ResourceVersionMatch
		prefix                     string
		pred                       storage.SelectionPredicate
		expectedOut                []*example.Pod
		expectContinue             bool
		expectedRemainingItemCount *int64
		expectError                bool
	}{
		{
			name:          "test List with limit when paging disabled",
			disablePaging: true,
			prefix:        "/two-level/",
			pred: storage.SelectionPredicate{
				Label: labels.Everything(),
				Field: fields.Everything(),
				Limit: 1,
			},
			expectedOut:    []*example.Pod{preset[1], preset[2]},
			expectContinue: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.pred.GetAttrs == nil {
				tt.pred.GetAttrs = getAttrs
			}

			out := &example.PodList{}
			storageOpts := storage.ListOptions{
				ResourceVersion:      tt.rv,
				ResourceVersionMatch: tt.rvMatch,
				Predicate:            tt.pred,
				Recursive:            true,
			}

			if err := store.GetList(ctx, tt.prefix, storageOpts, out); err != nil {
				t.Fatalf("GetList failed: %v", err)
				return
			}
			if (len(out.Continue) > 0) != tt.expectContinue {
				t.Errorf("unexpected continue token: %q", out.Continue)
			}

			if len(tt.expectedOut) != len(out.Items) {
				t.Fatalf("length of list want=%d, got=%d", len(tt.expectedOut), len(out.Items))
			}
			if diff := cmp.Diff(tt.expectedRemainingItemCount, out.ListMeta.GetRemainingItemCount()); diff != "" {
				t.Errorf("incorrect remainingItemCount: %s", diff)
			}
			for j, wantPod := range tt.expectedOut {
				getPod := &out.Items[j]
				ExpectNoDiff(t, fmt.Sprintf("%s: incorrect pod", tt.name), wantPod, getPod)
			}
		})
	}
}

// seedMultiLevelData creates a set of keys with a multi-level structure, returning a resourceVersion
// from before any were created along with the full set of objects that were persisted
func seedMultiLevelData(ctx context.Context, store storage.Interface) (string, []*example.Pod, error) {
	// Setup storage with the following structure:
	//  /
	//   - one-level/
	//  |            - test
	//  |
	//   - two-level/
	//  |            - 1/
	//  |           |   - test
	//  |           |
	//  |            - 2/
	//  |               - test
	//  |
	//   - z-level/
	//               - 3/
	//              |   - test
	//              |
	//               - 3/
	//                  - test-2
	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{
		{
			key: "/one-level/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/1/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/2/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
		},
		{
			key: "/z-level/3/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "fourth"}},
		},
		{
			key: "/z-level/3/test-2",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
		},
	}

	// we want to figure out the resourceVersion before we create anything
	initialList := &example.PodList{}
	if err := store.GetList(ctx, "/", storage.ListOptions{Predicate: storage.Everything, Recursive: true}, initialList); err != nil {
		return "", nil, fmt.Errorf("failed to determine starting resourceVersion: %w", err)
	}
	initialRV := initialList.ResourceVersion

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			return "", nil, fmt.Errorf("failed to create object: %w", err)
		}
	}

	var created []*example.Pod
	for _, item := range preset {
		created = append(created, item.storedObj)
	}
	return initialRV, created, nil
}

func RunTestGetListNonRecursive(ctx context.Context, t *testing.T, store storage.Interface) {
	prevKey, prevStoredObj := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "prev"}})

	prevRV, _ := strconv.Atoi(prevStoredObj.ResourceVersion)
	key, storedObj := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	currentRV, _ := strconv.Atoi(storedObj.ResourceVersion)

	tests := []struct {
		name             string
		key              string
		pred             storage.SelectionPredicate
		expectedOut      []*example.Pod
		rv               string
		rvMatch          metav1.ResourceVersionMatch
		expectRVTooLarge bool
	}{{
		name:        "existing key",
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
	}, {
		name:        "existing key, resourceVersion=0",
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          "0",
	}, {
		name:        "existing key, resourceVersion=0, resourceVersionMatch=notOlderThan",
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          "0",
		rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
	}, {
		name:        "existing key, resourceVersion=current",
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", currentRV),
	}, {
		name:        "existing key, resourceVersion=current, resourceVersionMatch=notOlderThan",
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", currentRV),
		rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
	}, {
		name:        "existing key, resourceVersion=previous, resourceVersionMatch=notOlderThan",
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", prevRV),
		rvMatch:     metav1.ResourceVersionMatchNotOlderThan,
	}, {
		name:        "existing key, resourceVersion=current, resourceVersionMatch=exact",
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
		rv:          fmt.Sprintf("%d", currentRV),
		rvMatch:     metav1.ResourceVersionMatchExact,
	}, {
		name:        "existing key, resourceVersion=current, resourceVersionMatch=exact",
		key:         prevKey,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{prevStoredObj},
		rv:          fmt.Sprintf("%d", prevRV),
		rvMatch:     metav1.ResourceVersionMatchExact,
	}, {
		name:             "existing key, resourceVersion=too high",
		key:              key,
		pred:             storage.Everything,
		expectedOut:      []*example.Pod{storedObj},
		rv:               strconv.FormatInt(math.MaxInt64, 10),
		expectRVTooLarge: true,
	}, {
		name:        "non-existing key",
		key:         "/non-existing",
		pred:        storage.Everything,
		expectedOut: nil,
	}, {
		name: "with matching pod name",
		key:  "/non-existing",
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("metadata.name!=" + storedObj.Name),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		},
		expectedOut: nil,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &example.PodList{}
			storageOpts := storage.ListOptions{
				ResourceVersion:      tt.rv,
				ResourceVersionMatch: tt.rvMatch,
				Predicate:            tt.pred,
				Recursive:            false,
			}
			err := store.GetList(ctx, tt.key, storageOpts, out)

			if tt.expectRVTooLarge {
				if err == nil || !storage.IsTooLargeResourceVersion(err) {
					t.Errorf("%s: expecting resource version too high error, but get: %s", tt.name, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("GetList failed: %v", err)
			}
			if len(out.ResourceVersion) == 0 {
				t.Errorf("%s: unset resourceVersion", tt.name)
			}
			if len(out.Items) != len(tt.expectedOut) {
				t.Errorf("%s: length of list want=%d, get=%d", tt.name, len(tt.expectedOut), len(out.Items))
				return
			}
			for j, wantPod := range tt.expectedOut {
				getPod := &out.Items[j]
				ExpectNoDiff(t, fmt.Sprintf("%s: incorrect pod", tt.name), wantPod, getPod)
			}
		})
	}
}

type CallsValidation func(t *testing.T, pageSize, estimatedProcessedObjects uint64)

func RunTestListContinuation(ctx context.Context, t *testing.T, store storage.Interface, validation CallsValidation) {
	// Setup storage with the following structure:
	//  /
	//   - one-level/
	//  |            - test
	//  |
	//   - two-level/
	//               - 1/
	//              |   - test
	//              |
	//               - 2/
	//                  - test
	//
	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{
		{
			key: "/one-level/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/1/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/2/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
		},
	}

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	// test continuations
	out := &example.PodList{}
	pred := func(limit int64, continueValue string) storage.SelectionPredicate {
		return storage.SelectionPredicate{
			Limit:    limit,
			Continue: continueValue,
			Label:    labels.Everything(),
			Field:    fields.Everything(),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		}
	}
	options := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, ""),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	ExpectNoDiff(t, "incorrect first page", []example.Pod{*preset[0].storedObj}, out.Items)
	if validation != nil {
		validation(t, 1, 1)
	}

	continueFromSecondItem := out.Continue

	// no limit, should get two items
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(0, continueFromSecondItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	key, rv, err := storage.DecodeContinue(continueFromSecondItem, "/")
	t.Logf("continue token was %d %s %v", rv, key, err)
	ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[1].storedObj, *preset[2].storedObj}, out.Items)
	if validation != nil {
		validation(t, 0, 2)
	}

	// limit, should get two more pages
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, continueFromSecondItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[1].storedObj}, out.Items)
	if validation != nil {
		validation(t, 1, 1)
	}

	continueFromThirdItem := out.Continue

	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, continueFromThirdItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	ExpectNoDiff(t, "incorrect third page", []example.Pod{*preset[2].storedObj}, out.Items)
	if validation != nil {
		validation(t, 1, 1)
	}
}

func RunTestListPaginationRareObject(ctx context.Context, t *testing.T, store storage.Interface, validation CallsValidation) {
	podCount := 1000
	var pods []*example.Pod
	for i := 0; i < podCount; i++ {
		key := fmt.Sprintf("/one-level/pod-%d", i)
		obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pod-%d", i)}}
		storedObj := &example.Pod{}
		err := store.Create(ctx, key, obj, storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
		pods = append(pods, storedObj)
	}

	out := &example.PodList{}
	options := storage.ListOptions{
		Predicate: storage.SelectionPredicate{
			Limit: 1,
			Label: labels.Everything(),
			Field: fields.OneTermEqualSelector("metadata.name", "pod-999"),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		},
		Recursive: true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Errorf("Unexpected continuation token set")
	}
	if len(out.Items) != 1 || !reflect.DeepEqual(&out.Items[0], pods[999]) {
		t.Fatalf("Unexpected first page: %#v", out.Items)
	}
	if validation != nil {
		validation(t, 1, uint64(podCount))
	}
}

func RunTestListContinuationWithFilter(ctx context.Context, t *testing.T, store storage.Interface, validation CallsValidation) {
	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{
		{
			key: "/1",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/2",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}, // this should not match
		},
		{
			key: "/3",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/4",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
	}

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	// the first list call should try to get 2 items from etcd (and only those items should be returned)
	// the field selector should result in it reading 3 items via the transformer
	// the chunking should result in 2 etcd Gets
	// there should be a continueValue because there is more data
	out := &example.PodList{}
	pred := func(limit int64, continueValue string) storage.SelectionPredicate {
		return storage.SelectionPredicate{
			Limit:    limit,
			Continue: continueValue,
			Label:    labels.Everything(),
			Field:    fields.OneTermNotEqualSelector("metadata.name", "bar"),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		}
	}
	options := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(2, ""),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Errorf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Errorf("No continuation token set")
	}
	ExpectNoDiff(t, "incorrect first page", []example.Pod{*preset[0].storedObj, *preset[2].storedObj}, out.Items)
	if validation != nil {
		validation(t, 2, 3)
	}

	// the rest of the test does not make sense if the previous call failed
	if t.Failed() {
		return
	}

	cont := out.Continue

	// the second list call should try to get 2 more items from etcd
	// but since there is only one item left, that is all we should get with no continueValue
	// both read counters should be incremented for the singular calls they make in this case
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(2, cont),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Errorf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Errorf("Unexpected continuation token set")
	}
	ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[3].storedObj}, out.Items)
	if validation != nil {
		validation(t, 2, 1)
	}
}

type Compaction func(ctx context.Context, t *testing.T, resourceVersion string)

func RunTestListInconsistentContinuation(ctx context.Context, t *testing.T, store storage.Interface, compaction Compaction) {
	if compaction == nil {
		t.Skipf("compaction callback not provided")
	}

	// Setup storage with the following structure:
	//  /
	//   - one-level/
	//  |            - test
	//  |
	//   - two-level/
	//               - 1/
	//              |   - test
	//              |
	//               - 2/
	//                  - test
	//
	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{
		{
			key: "/one-level/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/1/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			key: "/two-level/2/test",
			obj: &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
		},
	}

	for i, ps := range preset {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	pred := func(limit int64, continueValue string) storage.SelectionPredicate {
		return storage.SelectionPredicate{
			Limit:    limit,
			Continue: continueValue,
			Label:    labels.Everything(),
			Field:    fields.Everything(),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		}
	}

	out := &example.PodList{}
	options := storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, ""),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get initial list: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	ExpectNoDiff(t, "incorrect first page", []example.Pod{*preset[0].storedObj}, out.Items)

	continueFromSecondItem := out.Continue

	// update /two-level/2/test/bar
	oldName := preset[2].obj.Name
	newPod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: oldName,
			Labels: map[string]string{
				"state": "new",
			},
		},
	}
	if err := store.GuaranteedUpdate(ctx, preset[2].key, preset[2].storedObj, false, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return newPod, nil, nil
		}, newPod); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	// compact to latest revision.
	lastRVString := preset[2].storedObj.ResourceVersion
	compaction(ctx, t, lastRVString)

	// The old continue token should have expired
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(0, continueFromSecondItem),
		Recursive:       true,
	}
	err := store.GetList(ctx, "/", options, out)
	if err == nil {
		t.Fatalf("unexpected no error")
	}
	if !strings.Contains(err.Error(), "The provided continue parameter is too old ") {
		t.Fatalf("unexpected error message %v", err)
	}
	status, ok := err.(apierrors.APIStatus)
	if !ok {
		t.Fatalf("expect error of implements the APIStatus interface, got %v", reflect.TypeOf(err))
	}
	inconsistentContinueFromSecondItem := status.Status().ListMeta.Continue
	if len(inconsistentContinueFromSecondItem) == 0 {
		t.Fatalf("expect non-empty continue token")
	}

	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, inconsistentContinueFromSecondItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) == 0 {
		t.Fatalf("No continuation token set")
	}
	validateResourceVersion := ResourceVersionNotOlderThan(lastRVString)
	ExpectNoDiff(t, "incorrect second page", []example.Pod{*preset[1].storedObj}, out.Items)
	if err := validateResourceVersion(out.ResourceVersion); err != nil {
		t.Fatal(err)
	}
	continueFromThirdItem := out.Continue
	resolvedResourceVersionFromThirdItem := out.ResourceVersion
	out = &example.PodList{}
	options = storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred(1, continueFromThirdItem),
		Recursive:       true,
	}
	if err := store.GetList(ctx, "/", options, out); err != nil {
		t.Fatalf("Unable to get second page: %v", err)
	}
	if len(out.Continue) != 0 {
		t.Fatalf("Unexpected continuation token set")
	}
	ExpectNoDiff(t, "incorrect third page", []example.Pod{*preset[2].storedObj}, out.Items)
	if out.ResourceVersion != resolvedResourceVersionFromThirdItem {
		t.Fatalf("Expected list resource version to be %s, got %s", resolvedResourceVersionFromThirdItem, out.ResourceVersion)
	}
}

type PrefixTransformerModifier func(*PrefixTransformer) value.Transformer

type InterfaceWithPrefixTransformer interface {
	storage.Interface

	UpdatePrefixTransformer(PrefixTransformerModifier) func()
}

func RunTestConsistentList(ctx context.Context, t *testing.T, store InterfaceWithPrefixTransformer) {
	nextPod := func(index uint32) (string, *example.Pod) {
		key := fmt.Sprintf("pod-%d", index)
		obj := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: key,
				Labels: map[string]string{
					"even": strconv.FormatBool(index%2 == 0),
				},
			},
		}
		return key, obj
	}

	transformer := &reproducingTransformer{
		store:      store,
		nextObject: nextPod,
	}

	revertTransformer := store.UpdatePrefixTransformer(
		func(previousTransformer *PrefixTransformer) value.Transformer {
			transformer.wrapped = previousTransformer
			return transformer
		})
	defer revertTransformer()

	for i := 0; i < 5; i++ {
		if err := transformer.createObject(ctx); err != nil {
			t.Fatalf("failed to create object: %v", err)
		}
	}

	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		pod, ok := obj.(*example.Pod)
		if !ok {
			return nil, nil, fmt.Errorf("invalid object")
		}
		return labels.Set(pod.Labels), nil, nil
	}
	predicate := storage.SelectionPredicate{
		Label:    labels.Set{"even": "true"}.AsSelector(),
		GetAttrs: getAttrs,
		Limit:    4,
	}

	result1 := example.PodList{}
	options := storage.ListOptions{
		Predicate: predicate,
		Recursive: true,
	}
	if err := store.GetList(ctx, "/", options, &result1); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	// List objects from the returned resource version.
	options = storage.ListOptions{
		Predicate:            predicate,
		ResourceVersion:      result1.ResourceVersion,
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		Recursive:            true,
	}

	result2 := example.PodList{}
	if err := store.GetList(ctx, "/", options, &result2); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	ExpectNoDiff(t, "incorrect lists", result1, result2)

	// Now also verify the  ResourceVersionMatchNotOlderThan.
	options.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan

	result3 := example.PodList{}
	if err := store.GetList(ctx, "/", options, &result3); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	options.ResourceVersion = result3.ResourceVersion
	options.ResourceVersionMatch = metav1.ResourceVersionMatchExact

	result4 := example.PodList{}
	if err := store.GetList(ctx, "/", options, &result4); err != nil {
		t.Fatalf("failed to list objects: %v", err)
	}

	ExpectNoDiff(t, "incorrect lists", result3, result4)
}

func RunTestGuaranteedUpdate(ctx context.Context, t *testing.T, store InterfaceWithPrefixTransformer, validation KeyValidation) {
	key := "/foo"

	tests := []struct {
		name                string
		key                 string
		ignoreNotFound      bool
		precondition        *storage.Preconditions
		expectNotFoundErr   bool
		expectInvalidObjErr bool
		expectNoUpdate      bool
		transformStale      bool
		hasSelfLink         bool
	}{{
		name:                "non-existing key, ignoreNotFound=false",
		key:                 "/non-existing",
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   true,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, {
		name:                "non-existing key, ignoreNotFound=true",
		key:                 "/non-existing",
		ignoreNotFound:      true,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, {
		name:                "existing key",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, {
		name:                "same data",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, {
		name:                "same data, a selfLink",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
		hasSelfLink:         true,
	}, {
		name:                "same data, stale",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
		transformStale:      true,
	}, {
		name:                "UID match",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("A"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, {
		name:                "UID mismatch",
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("B"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: true,
		expectNoUpdate:      true,
	}}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, storeObj := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}})

			out := &example.Pod{}
			name := fmt.Sprintf("foo-%d", i)
			if tt.expectNoUpdate {
				name = storeObj.Name
			}

			if tt.transformStale {
				revertTransformer := store.UpdatePrefixTransformer(
					func(transformer *PrefixTransformer) value.Transformer {
						transformer.stale = true
						return transformer
					})
				defer revertTransformer()
			}

			version := storeObj.ResourceVersion
			err := store.GuaranteedUpdate(ctx, tt.key, out, tt.ignoreNotFound, tt.precondition,
				storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
					if tt.expectNotFoundErr && tt.ignoreNotFound {
						if pod := obj.(*example.Pod); pod.Name != "" {
							t.Errorf("%s: expecting zero value, but get=%#v", tt.name, pod)
						}
					}
					pod := *storeObj
					if tt.hasSelfLink {
						pod.SelfLink = "testlink"
					}
					pod.Name = name
					return &pod, nil
				}), nil)

			if tt.expectNotFoundErr {
				if err == nil || !storage.IsNotFound(err) {
					t.Errorf("%s: expecting not found error, but get: %v", tt.name, err)
				}
				return
			}
			if tt.expectInvalidObjErr {
				if err == nil || !storage.IsInvalidObj(err) {
					t.Errorf("%s: expecting invalid UID error, but get: %s", tt.name, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("%s: GuaranteedUpdate failed: %v", tt.name, err)
			}
			if out.ObjectMeta.Name != name {
				t.Errorf("%s: pod name want=%s, get=%s", tt.name, name, out.ObjectMeta.Name)
			}
			if out.SelfLink != "" {
				t.Errorf("%s: selfLink should not be set", tt.name)
			}

			// verify that kv pair is not empty after set and that the underlying data matches expectations
			validation(ctx, t, key)

			switch tt.expectNoUpdate {
			case true:
				if version != out.ResourceVersion {
					t.Errorf("%s: expect no version change, before=%s, after=%s", tt.name, version, out.ResourceVersion)
				}
			case false:
				if version == out.ResourceVersion {
					t.Errorf("%s: expect version change, but get the same version=%s", tt.name, version)
				}
			}
		})
	}
}

func RunTestGuaranteedUpdateWithTTL(ctx context.Context, t *testing.T, store storage.Interface) {
	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			ttl := uint64(1)
			return input, &ttl, nil
		}, nil)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, storage.ListOptions{ResourceVersion: out.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	TestCheckEventType(t, watch.Deleted, w)
}

func RunTestGuaranteedUpdateChecksStoredData(ctx context.Context, t *testing.T, store InterfaceWithPrefixTransformer) {
	input := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	// serialize input into etcd with data that would be normalized by a write -
	// in this case, leading whitespace
	revertTransformer := store.UpdatePrefixTransformer(
		func(transformer *PrefixTransformer) value.Transformer {
			transformer.prefix = []byte(string(transformer.prefix) + " ")
			return transformer
		})
	_, initial := TestPropagateStore(ctx, t, store, input)
	revertTransformer()

	// this update should write the canonical value to etcd because the new serialization differs
	// from the stored serialization
	input.ResourceVersion = initial.ResourceVersion
	out := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return input, nil, nil
		}, input)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if out.ResourceVersion == initial.ResourceVersion {
		t.Errorf("guaranteed update should have updated the serialized data, got %#v", out)
	}

	lastVersion := out.ResourceVersion

	// this update should not write to etcd because the input matches the stored data
	input = out
	out = &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return input, nil, nil
		}, input)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if out.ResourceVersion != lastVersion {
		t.Errorf("guaranteed update should have short-circuited write, got %#v", out)
	}

	revertTransformer = store.UpdatePrefixTransformer(
		func(transformer *PrefixTransformer) value.Transformer {
			transformer.stale = true
			return transformer
		})
	defer revertTransformer()

	// this update should write to etcd because the transformer reported stale
	err = store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			return input, nil, nil
		}, input)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if out.ResourceVersion == lastVersion {
		t.Errorf("guaranteed update should have written to etcd when transformer reported stale, got %#v", out)
	}
}

func RunTestGuaranteedUpdateWithConflict(ctx context.Context, t *testing.T, store storage.Interface) {
	key, _ := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	errChan := make(chan error, 1)
	var firstToFinish sync.WaitGroup
	var secondToEnter sync.WaitGroup
	firstToFinish.Add(1)
	secondToEnter.Add(1)

	go func() {
		err := store.GuaranteedUpdate(ctx, key, &example.Pod{}, false, nil,
			storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
				pod := obj.(*example.Pod)
				pod.Name = "foo-1"
				secondToEnter.Wait()
				return pod, nil
			}), nil)
		firstToFinish.Done()
		errChan <- err
	}()

	updateCount := 0
	err := store.GuaranteedUpdate(ctx, key, &example.Pod{}, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			if updateCount == 0 {
				secondToEnter.Done()
				firstToFinish.Wait()
			}
			updateCount++
			pod := obj.(*example.Pod)
			pod.Name = "foo-2"
			return pod, nil
		}), nil)
	if err != nil {
		t.Fatalf("Second GuaranteedUpdate error %#v", err)
	}
	if err := <-errChan; err != nil {
		t.Fatalf("First GuaranteedUpdate error %#v", err)
	}

	if updateCount != 2 {
		t.Errorf("Should have conflict and called update func twice")
	}
}

func RunTestGuaranteedUpdateWithSuggestionAndConflict(ctx context.Context, t *testing.T, store storage.Interface) {
	key, originalPod := TestPropagateStore(ctx, t, store, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})

	// First, update without a suggestion so originalPod is outdated
	updatedPod := &example.Pod{}
	err := store.GuaranteedUpdate(ctx, key, updatedPod, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			pod.Name = "foo-2"
			return pod, nil
		}),
		nil,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Second, update using the outdated originalPod as the suggestion. Return a conflict error when
	// passed originalPod, and make sure that SimpleUpdate is called a second time after a live lookup
	// with the value of updatedPod.
	sawConflict := false
	updatedPod2 := &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, updatedPod2, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			if pod.Name != "foo-2" {
				if sawConflict {
					t.Fatalf("unexpected second conflict")
				}
				sawConflict = true
				// simulated stale object - return a conflict
				return nil, apierrors.NewConflict(example.SchemeGroupVersion.WithResource("pods").GroupResource(), "name", errors.New("foo"))
			}
			pod.Name = "foo-3"
			return pod, nil
		}),
		originalPod,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if updatedPod2.Name != "foo-3" {
		t.Errorf("unexpected pod name: %q", updatedPod2.Name)
	}

	// Third, update using a current version as the suggestion.
	// Return an error and make sure that SimpleUpdate is NOT called a second time,
	// since the live lookup shows the suggestion was already up to date.
	attempts := 0
	updatedPod3 := &example.Pod{}
	err = store.GuaranteedUpdate(ctx, key, updatedPod3, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			pod := obj.(*example.Pod)
			if pod.Name != updatedPod2.Name || pod.ResourceVersion != updatedPod2.ResourceVersion {
				t.Errorf(
					"unexpected live object (name=%s, rv=%s), expected name=%s, rv=%s",
					pod.Name,
					pod.ResourceVersion,
					updatedPod2.Name,
					updatedPod2.ResourceVersion,
				)
			}
			attempts++
			return nil, fmt.Errorf("validation or admission error")
		}),
		updatedPod2,
	)
	if err == nil {
		t.Fatalf("expected error, got none")
	}
	if attempts != 1 {
		t.Errorf("expected 1 attempt, got %d", attempts)
	}
}

func RunTestTransformationFailure(ctx context.Context, t *testing.T, store InterfaceWithPrefixTransformer) {
	preset := []struct {
		key       string
		obj       *example.Pod
		storedObj *example.Pod
	}{{
		key: "/one-level/test",
		obj: &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			Spec:       DeepEqualSafePodSpec(),
		},
	}, {
		key: "/two-level/1/test",
		obj: &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "baz"},
			Spec:       DeepEqualSafePodSpec(),
		},
	}}
	for i, ps := range preset[:1] {
		preset[i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[:1][i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	// create a second resource with an invalid prefix
	revertTransformer := store.UpdatePrefixTransformer(
		func(transformer *PrefixTransformer) value.Transformer {
			return NewPrefixTransformer([]byte("otherprefix!"), false)
		})
	for i, ps := range preset[1:] {
		preset[1:][i].storedObj = &example.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[1:][i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}
	revertTransformer()

	// List should fail
	var got example.PodList
	storageOpts := storage.ListOptions{
		Predicate: storage.Everything,
		Recursive: true,
	}
	if err := store.GetList(ctx, "/", storageOpts, &got); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error %v", err)
	}

	// Get should fail
	if err := store.Get(ctx, preset[1].key, storage.GetOptions{}, &example.Pod{}); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	updateFunc := func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		return input, nil, nil
	}
	// GuaranteedUpdate without suggestion should return an error
	if err := store.GuaranteedUpdate(ctx, preset[1].key, &example.Pod{}, false, nil, updateFunc, nil); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	// GuaranteedUpdate with suggestion should return an error if we don't change the object
	if err := store.GuaranteedUpdate(ctx, preset[1].key, &example.Pod{}, false, nil, updateFunc, preset[1].obj); err == nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Delete fails with internal error.
	if err := store.Delete(ctx, preset[1].key, &example.Pod{}, nil, storage.ValidateAllObjectFunc, nil); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if err := store.Get(ctx, preset[1].key, storage.GetOptions{}, &example.Pod{}); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func RunTestCount(ctx context.Context, t *testing.T, store storage.Interface) {
	resourceA := "/foo.bar.io/abc"

	// resourceA is intentionally a prefix of resourceB to ensure that the count
	// for resourceA does not include any objects from resourceB.
	resourceB := fmt.Sprintf("%sdef", resourceA)

	resourceACountExpected := 5
	for i := 1; i <= resourceACountExpected; i++ {
		obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("foo-%d", i)}}

		key := fmt.Sprintf("%s/%d", resourceA, i)
		TestPropagateStoreWithKey(ctx, t, store, key, obj)
	}

	resourceBCount := 4
	for i := 1; i <= resourceBCount; i++ {
		obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("foo-%d", i)}}

		key := fmt.Sprintf("%s/%d", resourceB, i)
		TestPropagateStoreWithKey(ctx, t, store, key, obj)
	}

	resourceACountGot, err := store.Count(resourceA)
	if err != nil {
		t.Fatalf("store.Count failed: %v", err)
	}

	// count for resourceA should not include the objects for resourceB
	// even though resourceA is a prefix of resourceB.
	if int64(resourceACountExpected) != resourceACountGot {
		t.Fatalf("store.Count for resource %s: expected %d but got %d", resourceA, resourceACountExpected, resourceACountGot)
	}
}
