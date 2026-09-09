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

package coverage

import (
	"context"
	"flag"
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

var (
	coverageFlag    bool
	overrideEnabled *bool
	overrideMu      sync.RWMutex
)

func init() {
	flag.BoolVar(&coverageFlag, "storage-coverage", false, "Enable storage API state coverage tracking and reporting")
}

func SetEnabled(enabled *bool) {
	overrideMu.Lock()
	defer overrideMu.Unlock()
	overrideEnabled = enabled
}

func IsEnabled() bool {
	overrideMu.RLock()
	if overrideEnabled != nil {
		defer overrideMu.RUnlock()
		return *overrideEnabled
	}
	overrideMu.RUnlock()
	return coverageFlag
}

// CoverageStorage transparently wraps a storage.Interface to intercept and classify all invocations.
type CoverageStorage struct {
	storage.Interface
	tracker *CoverageTracker
}

func Wrap(store storage.Interface, tracker *CoverageTracker) storage.Interface {
	if !IsEnabled() || tracker == nil {
		return store
	}
	return &CoverageStorage{
		Interface: store,
		tracker:   tracker,
	}
}

// Tracker returns the underlying CoverageTracker.
func (s *CoverageStorage) Tracker() *CoverageTracker {
	if s == nil {
		return nil
	}
	return s.tracker
}

func (s *CoverageStorage) Versioner() storage.Versioner {
	return s.Interface.Versioner()
}

func (s *CoverageStorage) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	err := s.Interface.Create(ctx, key, obj, out, ttl)
	s.tracker.RecordCreate(key, obj, out, ttl, err)
	return err
}

func (s *CoverageStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	validateReject := false
	wrappedValidate := validateDeletion
	if validateDeletion != nil {
		wrappedValidate = func(ctx context.Context, obj runtime.Object) error {
			vErr := validateDeletion(ctx, obj)
			if vErr != nil {
				validateReject = true
			}
			return vErr
		}
	}

	err := s.Interface.Delete(ctx, key, out, preconditions, wrappedValidate, cachedExistingObject, opts)
	s.tracker.RecordDelete(key, preconditions, validateReject, err)
	return err
}

func (s *CoverageStorage) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	w, err := s.Interface.Watch(ctx, key, opts)
	s.tracker.RecordWatch(key, opts, err)
	if err != nil {
		return nil, err
	}
	return newCoverageWatcher(w, key, opts, s.tracker), nil
}

func (s *CoverageStorage) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	err := s.Interface.Get(ctx, key, opts, objPtr)
	s.tracker.RecordGet(key, opts, objPtr, err)
	return err
}

func (s *CoverageStorage) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	err := s.Interface.GetList(ctx, key, opts, listObj)
	s.tracker.RecordGetList(key, opts, listObj, err)
	return err
}

func (s *CoverageStorage) GuaranteedUpdate(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	isShortCircuit := false
	isMutating := false

	wrappedUpdate := func(existing runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		newObj, ttl, err := tryUpdate(existing, res)
		if err != nil {
			return nil, nil, err
		}
		if existing != nil && reflect.DeepEqual(existing, newObj) {
			isShortCircuit = true
		} else {
			isMutating = true
		}
		return newObj, ttl, nil
	}

	err := s.Interface.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, wrappedUpdate, cachedExistingObject)
	s.tracker.RecordUpdate(key, ignoreNotFound, preconditions, isMutating, isShortCircuit, cachedExistingObject != nil, err)
	return err
}

func (s *CoverageStorage) ReadinessCheck() error {
	return s.Interface.ReadinessCheck()
}
