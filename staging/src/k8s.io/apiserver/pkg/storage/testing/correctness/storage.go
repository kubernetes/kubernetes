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
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// ModelStorage implements storage.Interface on top of Model, so the storage
// compatibility scenarios (storagetesting.RunTest*) that etcd3 and cacher run
// can also be run against the model. It only translates calls into model
// requests, so passing scenarios validate the model itself. Like the model,
// it panics on options that aren't supported.
type ModelStorage struct {
	mu    sync.Mutex
	state *Model
}

var _ storage.Interface = (*ModelStorage)(nil)

// NewStorage returns a Storage starting from initialState.
func NewStorage(initialState *Model) *ModelStorage {
	return &ModelStorage{state: initialState.Clone()}
}

func (s *ModelStorage) Versioner() storage.Versioner {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state.Versioner
}

func (s *ModelStorage) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if ttl != 0 {
		panic("not implemented")
	}
	return s.execute(Request{
		Op:     OpCreate,
		Key:    key,
		Create: CreateRequest{Object: obj},
	}, out)
}

func (s *ModelStorage) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	if opts.ExpectTransformOrDecodeError {
		panic("not implemented")
	}
	return s.execute(Request{
		Op:  OpDelete,
		Key: key,
		Delete: DeleteRequest{
			Preconditions:        preconditions,
			ValidateDeletion:     validateDeletion,
			CachedExistingObject: cachedExistingObject,
		},
	}, out)
}

func (s *ModelStorage) GuaranteedUpdate(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	return s.execute(Request{
		Op:  OpUpdate,
		Key: key,
		Update: UpdateRequest{
			UpdateFunc:           tryUpdate,
			IgnoreNotFound:       ignoreNotFound,
			Preconditions:        preconditions,
			CachedExistingObject: cachedExistingObject,
		},
	}, destination)
}

func (s *ModelStorage) Get(ctx context.Context, key string, opts storage.GetOptions, out runtime.Object) error {
	return s.execute(Request{
		Op:  OpGet,
		Key: key,
		Get: GetRequest{Options: opts},
	}, out)
}

func (s *ModelStorage) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	return s.execute(Request{
		Op:   OpList,
		Key:  key,
		List: ListRequest{Options: opts},
	}, listObj)
}

func (s *ModelStorage) Compact(resourceVersion string) error {
	return s.execute(Request{
		Op:      OpCompact,
		Compact: CompactRequest{ResourceVersion: resourceVersion},
	}, nil)
}

func (s *ModelStorage) execute(req Request, out runtime.Object) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	resp, next, _ := s.state.execute(req)
	s.state = next
	if resp.Err != nil {
		return resp.Err
	}
	if out != nil {
		reflect.ValueOf(out).Elem().Set(reflect.ValueOf(resp.Object.DeepCopyObject()).Elem())
	}
	return nil
}

func (s *ModelStorage) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	panic("not implemented")
}

func (s *ModelStorage) Stats(ctx context.Context) (storage.Stats, error) {
	panic("not implemented")
}

func (s *ModelStorage) ReadinessCheck() error {
	panic("not implemented")
}

func (s *ModelStorage) RequestWatchProgress(ctx context.Context) error {
	panic("not implemented")
}

func (s *ModelStorage) GetCurrentResourceVersion(ctx context.Context) (uint64, error) {
	panic("not implemented")
}

func (s *ModelStorage) EnableResourceSizeEstimation(storage.KeysFunc) error {
	panic("not implemented")
}

func (s *ModelStorage) CompactRevision() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return int64(s.state.CompactResourceVersion)
}
