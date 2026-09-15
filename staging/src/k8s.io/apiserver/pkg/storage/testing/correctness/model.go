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

package correctness

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// NewModelFromStorage initializes a State from storage by listing all objects under prefix.
func NewModelFromStorage(prefix string, list runtime.Object, newFunc func() runtime.Object, keyFunc func(runtime.Object) (string, error)) (*Model, error) {
	state := NewEmptyModel(prefix, newFunc)
	accessor, err := meta.ListAccessor(list)
	if err != nil {
		return nil, err
	}
	var versioner = storage.APIObjectVersioner{}
	rvStr := accessor.GetResourceVersion()
	if len(rvStr) > 0 {
		state.ResourceVersion, err = versioner.ParseResourceVersion(rvStr)
		if err != nil {
			return nil, err
		}
	}
	objs, err := meta.ExtractList(list)
	if err != nil {
		return nil, err
	}
	for _, obj := range objs {
		key, err := keyFunc(obj)
		if err != nil {
			return nil, err
		}
		state.Items[key] = obj.DeepCopyObject()
	}
	return state, nil
}

// NewEmptyModel returns a new Model with no items.
func NewEmptyModel(prefix string, newFunc func() runtime.Object) *Model {
	return &Model{
		Prefix:          prefix,
		ResourceVersion: 1,
		Items:           make(map[string]runtime.Object),
		NewFunc:         newFunc,
	}
}

// Model is a state machine that mimics kubernetes storage behavior. Used for Linarizability testing of storage.
type Model struct {
	Items           map[string]runtime.Object
	ResourceVersion uint64
	Prefix          string
	NewFunc         func() runtime.Object
}

func (s *Model) Clone() *Model {
	clone := &Model{
		Items:           make(map[string]runtime.Object, len(s.Items)),
		ResourceVersion: s.ResourceVersion,
		Prefix:          s.Prefix,
		NewFunc:         s.NewFunc,
	}
	for k, v := range s.Items {
		if v != nil {
			clone.Items[k] = v.DeepCopyObject()
		}
	}
	return clone
}

func (s *Model) Equal(other *Model) bool {
	return s.ResourceVersion == other.ResourceVersion && s.Prefix == other.Prefix && reflect.DeepEqual(s.Items, other.Items)
}

// Step applies an operation to the sequential state machine. event is the watch
// event the operation produced, or nil if the operation didn't write.
func (s *Model) Step(input Request, output Response) (ok bool, next *Model, event *watch.Event) {
	next = s
	var expected Response
	switch input.Op {
	case OpCreate:
		next = s.Clone()
		expected, event = next.create(input.Key, input.Create.Object)
	case OpDelete:
		next = s.Clone()
		expected, event = next.delete(context.Background(), input.Key, input.Delete.Preconditions, nil)
	case OpGet:
		expected = s.get(input.Key, input.Get.Options)
	case OpUpdate:
		next = s.Clone()
		expected, event = next.update(context.Background(), input.Key, input.Update.IgnoreNotFound, input.Update.Preconditions, input.Update.UpdateFunc, input.Update.CachedExistingObject)
	}
	if !reflect.DeepEqual(expected, output) {
		return false, s, nil
	}
	return true, next, event
}

func (s *Model) update(ctx context.Context, key string, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) (Response, *watch.Event) {
	stored, exists := s.Items[key]
	var currentObj runtime.Object
	var currentRV uint64
	if !exists {
		if !ignoreNotFound {
			return Response{Object: nil, Err: storage.NewKeyNotFoundError(s.Prefix+key, int64(s.ResourceVersion))}, nil
		}
		if s.NewFunc == nil {
			return Response{Object: nil, Err: fmt.Errorf("NewFunc must be provided when ignoreNotFound=true")}, nil
		}
		currentObj = s.NewFunc()
		currentRV = 0
	} else {
		currentObj = stored.DeepCopyObject()
		currentRV = s.ResourceVersion
	}

	if err := preconditions.Check(s.Prefix+key, currentObj); err != nil {
		return Response{Object: nil, Err: err}, nil
	}

	if tryUpdate == nil {
		return Response{Object: nil, Err: fmt.Errorf("tryUpdate function must not be nil")}, nil
	}

	updated, _, err := tryUpdate(currentObj, storage.ResponseMeta{ResourceVersion: currentRV})
	if err != nil {
		return Response{Object: nil, Err: err}, nil
	}

	if exists {
		// Check for no-op update: compare data excluding ResourceVersion
		storedWithoutRV := stored.DeepCopyObject()
		if acc, err := meta.Accessor(storedWithoutRV); err == nil {
			acc.SetResourceVersion("")
		}
		updatedWithoutRV := updated.DeepCopyObject()
		if acc, err := meta.Accessor(updatedWithoutRV); err == nil {
			acc.SetResourceVersion("")
		}
		if reflect.DeepEqual(storedWithoutRV, updatedWithoutRV) {
			// Nothing is written, so no event is produced.
			return Response{Object: stored.DeepCopyObject(), Err: nil}, nil
		}
	}

	s.ResourceVersion++
	copied := updated.DeepCopyObject()
	accessor, err := meta.Accessor(copied)
	if err != nil {
		return Response{Object: nil, Err: err}, nil
	}
	accessor.SetResourceVersion(strconv.FormatUint(s.ResourceVersion, 10))
	s.Items[key] = copied
	// An update with ignoreNotFound on a missing key creates the object.
	eventType := watch.Modified
	if !exists {
		eventType = watch.Added
	}
	return Response{Object: copied, Err: nil}, &watch.Event{Type: eventType, Object: copied}
}

func (s *Model) create(key string, obj runtime.Object) (Response, *watch.Event) {
	if _, exists := s.Items[key]; exists {
		return Response{Object: nil, Err: storage.NewKeyExistsError(s.Prefix+key, 0)}, nil
	}
	s.ResourceVersion++
	copied := obj.DeepCopyObject()
	accessor, err := meta.Accessor(copied)
	if err != nil {
		return Response{Object: nil, Err: err}, nil
	}
	accessor.SetResourceVersion(strconv.FormatUint(s.ResourceVersion, 10))
	s.Items[key] = copied
	return Response{Object: copied, Err: nil}, &watch.Event{Type: watch.Added, Object: copied}
}

func (s *Model) get(key string, opts storage.GetOptions) Response {
	stored, exists := s.Items[key]
	if !exists {
		if opts.IgnoreNotFound {
			return Response{Object: nil, Err: nil}
		}
		return Response{Object: nil, Err: storage.NewKeyNotFoundError(s.Prefix+key, 0)}
	}
	return Response{Object: stored.DeepCopyObject(), Err: nil}
}

func (s *Model) delete(ctx context.Context, key string, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc) (Response, *watch.Event) {
	stored, exists := s.Items[key]
	if !exists {
		return Response{Object: nil, Err: storage.NewKeyNotFoundError(s.Prefix+key, int64(s.ResourceVersion))}, nil
	}
	if err := preconditions.Check(s.Prefix+key, stored); err != nil {
		return Response{Object: nil, Err: err}, nil
	}
	if validateDeletion != nil && stored != nil {
		if err := validateDeletion(ctx, stored); err != nil {
			return Response{Object: nil, Err: err}, nil
		}
	}
	s.ResourceVersion++
	deletedObj := stored.DeepCopyObject()
	accessor, err := meta.Accessor(deletedObj)
	if err != nil {
		return Response{Object: nil, Err: err}, nil
	}
	accessor.SetResourceVersion(strconv.FormatUint(s.ResourceVersion, 10))
	delete(s.Items, key)
	return Response{Object: deletedObj, Err: nil}, &watch.Event{Type: watch.Deleted, Object: deletedObj}
}

func (s *Model) Describe() string {
	items := []string{}
	for key, item := range s.Items {
		accessor, err := meta.Accessor(item)
		if err != nil {
			items = append(items, fmt.Sprintf("<p>%s: err: %v</p>", key, err))
		} else {
			items = append(items, fmt.Sprintf("<p>%s: %s, RV:%s</p>", key, accessor.GetUID(), accessor.GetResourceVersion()))
		}
	}
	return fmt.Sprintf("RV: %d, Items: %s", s.ResourceVersion, strings.Join(items, ""))
}
