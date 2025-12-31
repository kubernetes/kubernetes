/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

type DummyStorage struct {
	sync.RWMutex
	GetListErr error
	WatchErr   error
	GetListFn  func(_ context.Context, _ string, _ storage.ListOptions, listObj runtime.Object) error
	GetRVFn    func(_ context.Context) (uint64, error)
	WatchFn    func(_ context.Context, _ string, _ storage.ListOptions) (watch.Interface, error)

	// use GetRequestWatchProgressCounter when reading
	// the value of the counter
	RequestWatchProgressCounter int
}

func (d *DummyStorage) RequestWatchProgress(ctx context.Context) error {
	d.Lock()
	defer d.Unlock()
	d.RequestWatchProgressCounter++
	return nil
}

func (d *DummyStorage) GetRequestWatchProgressCounter() int {
	d.RLock()
	defer d.RUnlock()
	return d.RequestWatchProgressCounter
}

func (d *DummyStorage) CompactRevision() int64 {
	return 0
}

func (d *DummyStorage) IsWatchListSemanticsUnSupported() bool {
	return true
}

type DummyWatch struct {
	ch chan watch.Event
}

func (w *DummyWatch) ResultChan() <-chan watch.Event {
	return w.ch
}

func (w *DummyWatch) Stop() {
	close(w.ch)
}

func NewDummyWatch() watch.Interface {
	return &DummyWatch{
		ch: make(chan watch.Event),
	}
}

func (d *DummyStorage) Versioner() storage.Versioner { return nil }
func (d *DummyStorage) Create(_ context.Context, _ string, _, _ runtime.Object, _ uint64) error {
	return fmt.Errorf("unimplemented")
}
func (d *DummyStorage) Delete(_ context.Context, _ string, _ runtime.Object, _ *storage.Preconditions, _ storage.ValidateObjectFunc, _ runtime.Object, _ storage.DeleteOptions) error {
	return fmt.Errorf("unimplemented")
}
func (d *DummyStorage) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	if d.WatchFn != nil {
		return d.WatchFn(ctx, key, opts)
	}
	d.RLock()
	defer d.RUnlock()

	return NewDummyWatch(), d.WatchErr
}
func (d *DummyStorage) Get(_ context.Context, _ string, _ storage.GetOptions, _ runtime.Object) error {
	d.RLock()
	defer d.RUnlock()
	return d.GetListErr
}
func (d *DummyStorage) GetList(ctx context.Context, resPrefix string, opts storage.ListOptions, listObj runtime.Object) error {
	if d.GetListFn != nil {
		return d.GetListFn(ctx, resPrefix, opts, listObj)
	}
	d.RLock()
	defer d.RUnlock()
	podList := listObj.(*example.PodList)
	podList.ListMeta = metav1.ListMeta{ResourceVersion: "100"}
	return d.GetListErr
}
func (d *DummyStorage) GuaranteedUpdate(_ context.Context, _ string, _ runtime.Object, _ bool, _ *storage.Preconditions, _ storage.UpdateFunc, _ runtime.Object) error {
	return fmt.Errorf("unimplemented")
}
func (d *DummyStorage) Stats(_ context.Context) (storage.Stats, error) {
	return storage.Stats{}, fmt.Errorf("unimplemented")
}
func (d *DummyStorage) EnableResourceSizeEstimation(storage.KeysFunc) error {
	return nil
}
func (d *DummyStorage) ReadinessCheck() error {
	return nil
}
func (d *DummyStorage) InjectGetListError(err error) {
	d.Lock()
	defer d.Unlock()

	d.GetListErr = err
}

func (d *DummyStorage) GetCurrentResourceVersion(ctx context.Context) (uint64, error) {
	if d.GetRVFn != nil {
		return d.GetRVFn(ctx)
	}
	return 100, nil
}

type DummyCacher struct {
	DummyStorage
	IsReady    bool
	Consistent bool
}

func (d *DummyCacher) Ready() bool {
	return d.IsReady
}

func (d *DummyCacher) MarkConsistent(consistent bool) {
	d.Consistent = consistent
}
