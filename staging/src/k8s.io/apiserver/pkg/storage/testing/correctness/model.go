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
	"maps"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/storage"
)

// NewModelFromStorage initializes a State from storage by listing all objects under prefix.
func NewModelFromStorage(prefix string, list runtime.Object, newFunc, newListFunc func() runtime.Object, keyFunc func(runtime.Object) (string, error), versioner storage.Versioner) (*Model, error) {
	state := NewEmptyModel(prefix, newFunc, newListFunc, versioner)
	accessor, err := meta.ListAccessor(list)
	if err != nil {
		return nil, err
	}
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
func NewEmptyModel(prefix string, newFunc, newListFunc func() runtime.Object, versioner storage.Versioner) *Model {
	return &Model{
		Prefix:          prefix,
		ResourceVersion: 1,
		Items:           make(map[string]runtime.Object),
		NewFunc:         newFunc,
		NewListFunc:     newListFunc,
		Versioner:       versioner,
	}
}

// Model is a state machine that mimics kubernetes storage behavior. Used for Linarizability testing of storage.
type Model struct {
	Items           map[string]runtime.Object
	ResourceVersion uint64
	Prefix          string
	Versioner       storage.Versioner
	NewFunc         func() runtime.Object
	NewListFunc     func() runtime.Object
}

func (s *Model) Clone() *Model {
	clone := &Model{
		Items:           make(map[string]runtime.Object, len(s.Items)),
		ResourceVersion: s.ResourceVersion,
		Prefix:          s.Prefix,
		NewFunc:         s.NewFunc,
		NewListFunc:     s.NewListFunc,
		Versioner:       s.Versioner,
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

// Step applies an operation to the sequential state machine. change is the
// write the operation made, or nil if the operation didn't write.
func (s *Model) Step(input Request, output Response) (ok bool, next *Model, change *Change) {
	next = s
	var expected Response
	switch input.Op {
	case OpCreate:
		next = s.Clone()
		expected, change = next.create(input.Key, input.Create.Object)
	case OpDelete:
		next = s.Clone()
		expected, change = next.delete(context.Background(), input.Key, input.Delete.Preconditions, nil)
	case OpGet:
		expected = s.get(input.Key, input.Get.Options)
	case OpList:
		if err := checkKey(input.Key, input.List.Options.Recursive); err != nil {
			expected = Response{Err: err}
			break
		}
		if _, _, err := storage.ValidateListOptions("", s.Versioner, input.List.Options); err != nil {
			expected = Response{Err: err}
			break
		}
		if input.List.Options.Predicate.Label == nil || input.List.Options.Predicate.Field == nil {
			// etcd3 and the cacher call methods on both selectors.
			panic("nil label or field selector is not supported, use storage.Everything to match everything")
		}
		if input.List.Options.Predicate.Limit != 0 || input.List.Options.Predicate.Continue != "" {
			panic("pagination (limit, continue) is not supported")
		}
		if !input.List.Options.Predicate.Empty() && input.List.Options.Predicate.GetAttrs == nil {
			panic("selectors without GetAttrs are not supported")
		}
		if input.List.Options.RecordTimestamps {
			panic("recordTimestamps is not supported, it wraps objects in storage-internal types")
		}
		if !input.List.Options.Predicate.Empty() {
			panic("label and field selectors are not supported, the model doesn't filter lists")
		}
		// Model only validates consistent reads, for stale reads we just validate RV and contents are validated later during replay.
		if input.List.Options.ResourceVersion != "" {
			return s.validateListRV(input.List.Options, output), s, nil
		}
		expected = s.list(input.Key, input.List.Options)
	case OpUpdate:
		next = s.Clone()
		expected, change = next.update(context.Background(), input.Key, input.Update.IgnoreNotFound, input.Update.Preconditions, input.Update.UpdateFunc, input.Update.CachedExistingObject)
	default:
		panic(fmt.Sprintf("unknown operation %q", input.Op))
	}
	if !reflect.DeepEqual(expected, output) {
		return false, s, nil
	}
	return true, next, change
}

func (s *Model) validateListRV(opts storage.ListOptions, output Response) bool {
	if opts.ResourceVersion == "" {
		return true
	}
	reqRV, err := s.Versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return false
	}
	if output.Err != nil {
		// RV from future must return error
		if storage.IsTooLargeResourceVersion(output.Err) {
			return output.Object == nil && reqRV > s.ResourceVersion
		}
		return false
	}
	if output.Object == nil {
		return false
	}
	accessor, err := meta.ListAccessor(output.Object)
	if err != nil {
		return false
	}
	respRV, err := s.Versioner.ParseResourceVersion(accessor.GetResourceVersion())
	if err != nil || respRV == 0 {
		return false
	}
	// RV from future that didn't return error is invalid.
	if respRV > s.ResourceVersion {
		return false
	}
	switch opts.ResourceVersionMatch {
	case metav1.ResourceVersionMatchExact:
		return reqRV > 0 && respRV == reqRV
	case metav1.ResourceVersionMatchNotOlderThan:
		return respRV >= reqRV
	case "":
		// Legacy exact match
		if opts.Recursive && opts.Predicate.Limit > 0 && reqRV > 0 {
			return respRV == reqRV
		}
		return respRV >= reqRV
	default:
		return false
	}
}

func (s *Model) update(ctx context.Context, key string, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) (Response, *Change) {
	if err := checkKey(key, false); err != nil {
		return Response{Err: err}, nil
	}
	if tryUpdate == nil {
		panic("tryUpdate must be set")
	}
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
		rv, err := s.Versioner.ObjectResourceVersion(stored)
		if err != nil {
			return Response{Object: nil, Err: err}, nil
		}
		currentRV = rv
	}

	if err := preconditions.Check(s.Prefix+key, currentObj); err != nil {
		return Response{Object: nil, Err: err}, nil
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
	return Response{Object: copied, Err: nil}, &Change{Key: key, ResourceVersion: s.ResourceVersion, Object: copied, PrevObject: stored}
}

func (s *Model) create(key string, obj runtime.Object) (Response, *Change) {
	if err := checkKey(key, false); err != nil {
		return Response{Err: err}, nil
	}
	if rv, err := s.Versioner.ObjectResourceVersion(obj); err == nil && rv != 0 {
		return Response{Err: storage.ErrResourceVersionSetOnCreate}, nil
	}
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
	return Response{Object: copied, Err: nil}, &Change{Key: key, ResourceVersion: s.ResourceVersion, Object: copied}
}

func (s *Model) get(key string, opts storage.GetOptions) Response {
	if err := checkKey(key, false); err != nil {
		return Response{Err: err}
	}
	if opts.ResourceVersion != "" {
		panic("get with resourceVersion is not supported, the model only serves the latest state")
	}
	stored, exists := s.Items[key]
	if !exists {
		if opts.IgnoreNotFound {
			return Response{Object: s.NewFunc(), Err: nil}
		}
		return Response{Object: nil, Err: storage.NewKeyNotFoundError(s.Prefix+key, 0)}
	}
	return Response{Object: stored.DeepCopyObject(), Err: nil}
}

func (s *Model) list(key string, opts storage.ListOptions) Response {
	var items []runtime.Object
	for _, k := range slices.Sorted(maps.Keys(s.Items)) {
		if keyInScope(key, opts.Recursive, k) {
			items = append(items, s.Items[k].DeepCopyObject())
		}
	}
	list := s.NewListFunc()
	if err := meta.SetList(list, items); err != nil {
		return Response{Object: nil, Err: err}
	}
	if err := s.Versioner.UpdateList(list, s.ResourceVersion, "", nil); err != nil {
		return Response{Object: nil, Err: err}
	}
	return Response{Object: list, Err: nil}
}

// keyInScope reports whether k is selected by a list or watch on key.
func keyInScope(key string, recursive bool, k string) bool {
	if !recursive {
		return k == key
	}
	// Like storage.PrepareKey, "/a" must not select "/ab".
	return strings.HasPrefix(k, strings.TrimSuffix(key, "/")+"/")
}

func checkKey(key string, recursive bool) error {
	_, err := storage.PrepareKey("", key, recursive)
	return err
}

func (s *Model) delete(ctx context.Context, key string, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc) (Response, *Change) {
	if err := checkKey(key, false); err != nil {
		return Response{Err: err}, nil
	}
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
	return Response{Object: deletedObj, Err: nil}, &Change{Key: key, ResourceVersion: s.ResourceVersion, PrevObject: stored}
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
