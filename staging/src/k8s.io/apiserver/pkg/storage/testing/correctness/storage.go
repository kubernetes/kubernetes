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
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
)

const modelWatchResultBufferSize = 100

// ModelStorage implements storage.Interface by synchronizing access to Replay
// and streaming its recorded history to live watchers, so the storage
// compatibility scenarios (storagetesting.RunTest*) that etcd3 and cacher run
// can also be run against the model. Like the model, it panics on options that
// aren't supported.
type ModelStorage struct {
	mu       sync.Mutex
	replay   *Replay
	watchers map[*modelWatcher]struct{}
}

var _ storage.Interface = (*ModelStorage)(nil)

// NewStorage returns a Storage starting from initialState.
func NewStorage(initialState *Model) *ModelStorage {
	replay, err := NewReplay(initialState, nil)
	if err != nil {
		panic(err)
	}
	return &ModelStorage{
		replay:   replay,
		watchers: make(map[*modelWatcher]struct{}),
	}
}

func (s *ModelStorage) Versioner() storage.Versioner {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.replay.state.Versioner
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
	resp, change := s.replay.execute(req)
	if change != nil {
		for w := range s.watchers {
			select {
			case w.notifyChanges <- struct{}{}:
			default:
			}
		}
	}
	if resp.Err != nil {
		return resp.Err
	}
	if out != nil {
		reflect.ValueOf(out).Elem().Set(reflect.ValueOf(resp.Object.DeepCopyObject()).Elem())
	}
	return nil
}

func (s *ModelStorage) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	req := WatchRequest{Key: key, Options: opts}
	s.mu.Lock()
	initialEvents, nextRV, closed, err := s.replay.watch(req, s.replay.ResourceVersion())
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	w := &modelWatcher{
		resultChan:    make(chan watch.Event, modelWatchResultBufferSize),
		notifyChanges: make(chan struct{}, 1),
		stopWatcher:   make(chan struct{}),
	}
	if !closed && ctx.Err() == nil {
		s.watchers[w] = struct{}{}
	}
	s.mu.Unlock()

	utilflowcontrol.WatchInitialized(ctx)
	go w.run(ctx, s, req, initialEvents, nextRV, closed)
	return w, nil
}

func (s *ModelStorage) removeWatcher(w *modelWatcher) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.watchers, w)
}

type modelWatcher struct {
	resultChan    chan watch.Event
	notifyChanges chan struct{}
	stopWatcher   chan struct{}
	stopOnce      sync.Once
}

func (w *modelWatcher) Stop() {
	w.stopOnce.Do(func() {
		close(w.stopWatcher)
	})
}

func (w *modelWatcher) ResultChan() <-chan watch.Event {
	return w.resultChan
}

func (w *modelWatcher) send(ctx context.Context, ev watch.Event) bool {
	// Check cancellation before sending so an already-stopped or canceled
	// watcher never delivers an extra event into its buffered result channel.
	select {
	case <-ctx.Done():
		return false
	case <-w.stopWatcher:
		return false
	default:
	}
	select {
	case w.resultChan <- ev:
		return true
	case <-ctx.Done():
		return false
	case <-w.stopWatcher:
		return false
	}
}

func (w *modelWatcher) run(ctx context.Context, s *ModelStorage, req WatchRequest, initialEvents []watch.Event, nextRV uint64, closed bool) {
	defer func() {
		s.removeWatcher(w)
		close(w.resultChan)
	}()
	for _, ev := range initialEvents {
		if !w.send(ctx, ev) {
			return
		}
	}
	if closed {
		return
	}
	for {
		s.mu.Lock()
		currentRV := s.replay.ResourceVersion()
		var events []watch.Event
		var err error
		if nextRV <= currentRV {
			events, err = s.replay.EventsForRVs(req, &ResourceVersionRange{Min: nextRV, Max: currentRV + 1})
			nextRV = currentRV + 1
		}
		s.mu.Unlock()

		for _, ev := range events {
			if !w.send(ctx, ev) {
				return
			}
		}
		if err != nil {
			w.send(ctx, errorWatchEvent(err))
			return
		}

		select {
		case <-w.notifyChanges:
		case <-ctx.Done():
			return
		case <-w.stopWatcher:
			return
		}
	}
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
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.replay.ResourceVersion(), nil
}

func (s *ModelStorage) EnableResourceSizeEstimation(storage.KeysFunc) error {
	panic("not implemented")
}

func (s *ModelStorage) CompactRevision() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return int64(s.replay.state.CompactResourceVersion)
}

func (s *ModelStorage) DeleteOnce(ctx context.Context, key string, origState *storage.ObjState, out runtime.Object) error {
	panic("not implemented")
}

func (s *ModelStorage) UpdateOnce(ctx context.Context, key string, origState *storage.ObjState, obj runtime.Object, ttl uint64, destination runtime.Object, ignoreNotFound bool) (*storage.ObjState, error) {
	panic("not implemented")
}
