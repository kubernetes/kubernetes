/*
Copyright 2024 The Kubernetes Authors.

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

package registry

import (
	"context"
	"errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	storeerr "k8s.io/apiserver/pkg/storage/errors"

	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

// the corrupt object deleter has the same interface as rest.GracefulDeleter
var _ rest.GracefulDeleter = &corruptObjectDeleter{}

// NewCorruptObjectDeleter returns a deleter that can perform unsafe deletion of corrupt
// objects. The deletion will be rejected if the object turns out to be decodable (i.e. not actually
// corrupt).
//
// NOTE: it skips precondition checks, finalizer constraints, and any post deletion hook defined in
// 'AfterDelete' of the registry.
//
// WARNING: This may break the cluster if the resource being deleted has dependencies.
func NewCorruptObjectDeleter(store *Store) rest.GracefulDeleter {
	return &corruptObjectDeleter{store: store}
}

// corruptObjectDeleter implements unsafe object deletion flow
type corruptObjectDeleter struct {
	store *Store
}

// Delete performs an unsafe deletion of the given resource from the storage.
//
// NOTE: This function should NEVER be used for any normal deletion
// flow, it is exclusively used when the user enables
// 'IgnoreStoreReadErrorWithClusterBreakingPotential' in the delete options.
func (d *corruptObjectDeleter) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, opts *metav1.DeleteOptions) (runtime.Object, bool, error) {
	if opts == nil || !ptr.Deref[bool](opts.IgnoreStoreReadErrorWithClusterBreakingPotential, false) {
		// this is a developer error, we should never be here, since the unsafe
		// deleter is wired in the rest layer only when the option is enabled
		return nil, false, apierrors.NewInternalError(errors.New("initialization error, expected normal deletion flow to be used"))
	}

	key, err := d.store.KeyFunc(ctx, name)
	if err != nil {
		return nil, false, err
	}
	qualifiedResource := d.store.qualifiedResourceFromContext(ctx)
	// use the storage implementation directly, bypass the dryRun layer
	storageBackend := d.store.Storage.Storage
	klog.FromContext(ctx).V(1).Info("Going to perform unsafe object deletion", "object", klog.KRef(genericapirequest.NamespaceValue(ctx), name))
	out := d.store.NewFunc()
	storageOpts := storage.DeleteOptions{ExpectTransformOrDecodeError: true}
	// we don't have the old object in the cache, neither can it be
	// retrieved from the storage and decoded into an object
	// successfully, so we do the following:
	//  a) skip preconditions check
	//  b) skip admission validation, rest.ValidateAllObjectFunc will "admit everything"
	var nilPreconditions *storage.Preconditions = nil
	var nilCachedExistingObject runtime.Object = nil
	if err := storageBackend.Delete(ctx, key, out, nilPreconditions, rest.ValidateAllObjectFunc, nilCachedExistingObject, storageOpts); err != nil {
		return nil, false, storeerr.InterpretDeleteError(err, qualifiedResource, name)
	}
	// the DELETE succeeded, but we don't have the object since it's
	// not retrievable from the storage, so we send a nil object
	return nil, true, nil
}
