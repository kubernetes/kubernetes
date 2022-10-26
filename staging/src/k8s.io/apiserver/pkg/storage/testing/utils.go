/*
Copyright 2015 The Kubernetes Authors.

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
	"bytes"
	"context"
	"fmt"
	"path"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/value"
)

// CreateObj will create a single object using the storage interface.
func CreateObj(helper storage.Interface, name string, obj, out runtime.Object, ttl uint64) error {
	return helper.Create(context.TODO(), name, obj, out, ttl)
}

// CreateObjList will create a list from the array of objects.
func CreateObjList(prefix string, helper storage.Interface, items []runtime.Object) error {
	for i := range items {
		obj := items[i]
		meta, err := meta.Accessor(obj)
		if err != nil {
			return err
		}
		err = CreateObj(helper, path.Join(prefix, meta.GetName()), obj, obj, 0)
		if err != nil {
			return err
		}
		items[i] = obj
	}
	return nil
}

// CreateList will properly create a list using the storage interface.
func CreateList(prefix string, helper storage.Interface, list runtime.Object) error {
	items, err := meta.ExtractList(list)
	if err != nil {
		return err
	}
	err = CreateObjList(prefix, helper, items)
	if err != nil {
		return err
	}
	return meta.SetList(list, items)
}

// DeepEqualSafePodSpec returns an example.PodSpec safe for deep-equal operations.
func DeepEqualSafePodSpec() example.PodSpec {
	grace := int64(30)
	return example.PodSpec{
		RestartPolicy:                 "Always",
		TerminationGracePeriodSeconds: &grace,
		SchedulerName:                 "default-scheduler",
	}
}

// TestPropagateStore helps propagates store with objects, automates key generation, and returns
// keys and stored objects.
func TestPropagateStore(ctx context.Context, t *testing.T, store storage.Interface, obj *example.Pod) (string, *example.Pod) {
	// Setup store with a key and grab the output for returning.
	key := "/testkey"
	return key, TestPropagateStoreWithKey(ctx, t, store, key, obj)
}

// TestPropagateStoreWithKey helps propagate store with objects, the given object will be stored at the specified key.
func TestPropagateStoreWithKey(ctx context.Context, t *testing.T, store storage.Interface, key string, obj *example.Pod) *example.Pod {
	// Setup store with the specified key and grab the output for returning.
	err := store.Delete(ctx, key, &example.Pod{}, nil, storage.ValidateAllObjectFunc, nil)
	if err != nil && !storage.IsNotFound(err) {
		t.Fatalf("Cleanup failed: %v", err)
	}
	setOutput := &example.Pod{}
	if err := store.Create(ctx, key, obj, setOutput, 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	return setOutput
}

func ExpectNoDiff(t *testing.T, msg string, expected, got interface{}) {
	t.Helper()
	if !reflect.DeepEqual(expected, got) {
		if diff := cmp.Diff(expected, got); diff != "" {
			t.Errorf("%s: %s", msg, diff)
		} else {
			t.Errorf("%s:\nexpected: %#v\ngot: %#v", msg, expected, got)
		}
	}
}

const dummyPrefix = "adapter"

func EncodeContinueOrDie(key string, resourceVersion int64) string {
	token, err := storage.EncodeContinue(dummyPrefix+key, dummyPrefix, resourceVersion)
	if err != nil {
		panic(err)
	}
	return token
}

func TestCheckEventType(t *testing.T, expectEventType watch.EventType, w watch.Interface) {
	select {
	case res := <-w.ResultChan():
		if res.Type != expectEventType {
			t.Errorf("event type want=%v, get=%v", expectEventType, res.Type)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("time out after waiting %v on ResultChan", wait.ForeverTestTimeout)
	}
}

func TestCheckResult(t *testing.T, expectEventType watch.EventType, w watch.Interface, expectObj *example.Pod) {
	TestCheckResultFunc(t, expectEventType, w, func(object runtime.Object) error {
		ExpectNoDiff(t, "incorrect object", expectObj, object)
		return nil
	})
}

func TestCheckResultFunc(t *testing.T, expectEventType watch.EventType, w watch.Interface, check func(object runtime.Object) error) {
	select {
	case res := <-w.ResultChan():
		if res.Type != expectEventType {
			t.Errorf("event type want=%v, get=%v", expectEventType, res.Type)
			return
		}
		if err := check(res.Object); err != nil {
			t.Error(err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("time out after waiting %v on ResultChan", wait.ForeverTestTimeout)
	}
}

func TestCheckStop(t *testing.T, w watch.Interface) {
	select {
	case e, ok := <-w.ResultChan():
		if ok {
			var obj string
			switch e.Object.(type) {
			case *example.Pod:
				obj = e.Object.(*example.Pod).Name
			case *v1.Status:
				obj = e.Object.(*v1.Status).Message
			}
			t.Errorf("ResultChan should have been closed. Event: %s. Object: %s", e.Type, obj)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("time out after waiting 1s on ResultChan")
	}
}

// ResourceVersionNotOlderThan returns a function to validate resource versions. Resource versions
// referring to points in logical time before the sentinel generate an error. All logical times as
// new as the sentinel or newer generate no error.
func ResourceVersionNotOlderThan(sentinel string) func(string) error {
	return func(resourceVersion string) error {
		objectVersioner := storage.APIObjectVersioner{}
		actualRV, err := objectVersioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			return err
		}
		expectedRV, err := objectVersioner.ParseResourceVersion(sentinel)
		if err != nil {
			return err
		}
		if actualRV < expectedRV {
			return fmt.Errorf("expected a resourceVersion no smaller than than %d, but got %d", expectedRV, actualRV)
		}
		return nil
	}
}

// PrefixTransformer adds and verifies that all data has the correct prefix on its way in and out.
type PrefixTransformer struct {
	prefix []byte
	stale  bool
	err    error
	reads  uint64
}

func NewPrefixTransformer(prefix []byte, stale bool) *PrefixTransformer {
	return &PrefixTransformer{
		prefix: prefix,
		stale:  stale,
	}
}

func (p *PrefixTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	atomic.AddUint64(&p.reads, 1)
	if dataCtx == nil {
		panic("no context provided")
	}
	if !bytes.HasPrefix(data, p.prefix) {
		return nil, false, fmt.Errorf("value does not have expected prefix %q: %s,", p.prefix, string(data))
	}
	return bytes.TrimPrefix(data, p.prefix), p.stale, p.err
}
func (p *PrefixTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	if dataCtx == nil {
		panic("no context provided")
	}
	if len(data) > 0 {
		return append(append([]byte{}, p.prefix...), data...), p.err
	}
	return data, p.err
}

func (p *PrefixTransformer) GetReadsAndReset() uint64 {
	return atomic.SwapUint64(&p.reads, 0)
}
