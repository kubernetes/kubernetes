/*
Copyright The Kubernetes Authors.

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

package etcd3

import (
	"context"
	"errors"
	"fmt"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestAggregatedStorageErrorMessage(t *testing.T) {
	tests := []struct {
		name     string
		abortErr error
		want     string
	}{
		{
			name: "aggregation completed without an abort",
			want: "unable to transform or decode 2 objects: {\n\tfoo is corrupt\n\tbar is corrupt\n}",
		},
		{
			name:     "aggregation aborted after reaching the limit",
			abortErr: errTooMany,
			want:     "unable to transform or decode 2 objects: {\n\tfoo is corrupt\n\tbar is corrupt\n}, aborted: too many errors, the list is truncated",
		},
		{
			name:     "aggregation aborted due to an unexpected error",
			abortErr: errors.New("etcd is down"),
			want:     "unable to transform or decode 2 objects: {\n\tfoo is corrupt\n\tbar is corrupt\n}, aborted: etcd is down",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := &aggregatedStorageError{
				resourcePrefix: "list",
				errs:           utilerrors.NewAggregate([]error{errors.New("foo is corrupt"), errors.New("bar is corrupt")}),
				abortErr:       test.abortErr,
			}
			if want, got := test.want, err.Error(); want != got {
				t.Errorf("unexpected error message, want:\n%s\ngot:\n%s", want, got)
			}
		})
	}
}

// dataCorruptionError tells value.IsDataCorruption whether the stored bytes are
// damaged, as a storage compression read error does.
type dataCorruptionError bool

func (e dataCorruptionError) Error() string {
	return fmt.Sprintf("failed to transform, data corruption: %t", bool(e))
}

func (e dataCorruptionError) DataCorruption() bool { return bool(e) }

// failingReads fails every read with err.
type failingReads struct {
	value.Transformer
	err error
}

func (f *failingReads) TransformFromStorage(context.Context, []byte, value.Context) ([]byte, bool, error) {
	return nil, false, f.err
}

// TestOnlyDataCorruptionIsOfferedForUnsafeDeletion checks that an object whose
// read fails with an error marked as not data corruption isn't reported as
// corrupt and can't be deleted unsafely. An error with no DataCorruption method
// counts as corruption, as KEP-3926 treats every failed transform.
func TestOnlyDataCorruptionIsOfferedForUnsafeDeletion(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowUnsafeMalformedObjectDeletion, true)
	tests := []struct {
		name    string
		readErr error
		corrupt bool
	}{
		{name: "no DataCorruption method", readErr: errors.New("cipher: message authentication failed"), corrupt: true},
		{name: "data corruption", readErr: fmt.Errorf("wrapped: %w", dataCorruptionError(true)), corrupt: true},
		{name: "not data corruption", readErr: fmt.Errorf("wrapped: %w", dataCorruptionError(false)), corrupt: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, s, client := testSetup(t)
			key := "/pods/ns/foo"
			pod := &example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "foo"}}
			if err := s.Create(ctx, key, pod, nil, 0); err != nil {
				t.Fatal(err)
			}
			failing := WithCorruptObjErrorHandlingTransformer(&failingReads{Transformer: s.transformer, err: tc.readErr})
			s.transformer, s.watcher.transformer = failing, failing
			unsafeDeletionStore := NewStoreWithUnsafeCorruptObjectDeletion(s, s.groupResource)

			err := unsafeDeletionStore.Get(ctx, key, storage.GetOptions{}, &example.Pod{})
			if got := storage.IsCorruptObject(err); got != tc.corrupt {
				t.Errorf("Get: corrupt object = %t, want %t: %v", got, tc.corrupt, err)
			}
			// A list aggregates corrupt objects but stops at any other error.
			err = unsafeDeletionStore.GetList(ctx, "/pods/", storage.ListOptions{Predicate: storage.Everything, Recursive: true}, &example.PodList{})
			if got := apierrors.IsStoreReadError(err); got != tc.corrupt {
				t.Errorf("GetList: corrupt object = %t, want %t: %v", got, tc.corrupt, err)
			}

			err = s.Delete(ctx, key, &example.Pod{}, nil, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{ExpectTransformOrDecodeError: true})
			if tc.corrupt {
				if err != nil {
					t.Errorf("unsafe Delete of a corrupt object failed: %v", err)
				}
			} else if !storage.IsInternalError(err) || !errors.Is(err, tc.readErr) {
				t.Errorf("unsafe Delete: got %v, want an internal error wrapping %v", err, tc.readErr)
			}
			resp, err := client.KV.Get(ctx, key)
			if err != nil {
				t.Fatal(err)
			}
			if remains := len(resp.Kvs) == 1; remains == tc.corrupt {
				t.Errorf("after unsafe Delete, key still in etcd = %t, want %t", remains, !tc.corrupt)
			}
		})
	}
}
