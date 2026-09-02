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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// wrapped decorates a storage.Interface, classifying Get/GetList/Watch/Delete
// calls into the Recorder and delegating every method - including these four
// - unchanged to the wrapped Interface. Embedding storage.Interface means any
// method this file does not override still satisfies the interface exactly
// as the wrapped implementation defines it, so Wrap stays non-invasive as
// storage.Interface grows.
type wrapped struct {
	storage.Interface
	rec *Recorder
}

var _ storage.Interface = (*wrapped)(nil)

// Wrap returns a storage.Interface that behaves exactly like inner, while
// recording the API state of every Get, GetList, Watch and Delete call into
// rec. rec may be shared across multiple Wrap-ed instances to aggregate
// coverage over a whole test suite.
func Wrap(inner storage.Interface, rec *Recorder) storage.Interface {
	return &wrapped{Interface: inner, rec: rec}
}

func (w *wrapped) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	err := w.Interface.Get(ctx, key, opts, objPtr)
	w.rec.Observe(ClassifyGet(opts, err))
	return err
}

func (w *wrapped) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	err := w.Interface.GetList(ctx, key, opts, listObj)
	w.rec.Observe(ClassifyList(opts, err))
	return err
}

func (w *wrapped) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	wi, err := w.Interface.Watch(ctx, key, opts)
	w.rec.Observe(ClassifyWatch(opts, err))
	return wi, err
}

func (w *wrapped) Delete(
	ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions,
	validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions,
) error {
	err := w.Interface.Delete(ctx, key, out, preconditions, validateDeletion, cachedExistingObject, opts)
	w.rec.Observe(ClassifyDelete(preconditions, err))
	return err
}
