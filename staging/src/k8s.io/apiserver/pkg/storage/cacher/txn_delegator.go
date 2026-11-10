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

package cacher

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/klog/v2"
)

type txnDelegator struct {
	cacher  *Cacher
	storage txnStorage
	key     string
}

// txnStorage provides subset of storage.Interface with just Txn operations needed for delegator.
type txnStorage interface {
	DeleteOnce(ctx context.Context, key string, origState *storage.ObjState, out runtime.Object) error
	UpdateOnce(ctx context.Context, key string, origState *storage.ObjState, obj runtime.Object, ttl uint64, destination runtime.Object, ignoreNotFound bool) (*storage.ObjState, error)
}

var _ storage.TxnDelegator = (*txnDelegator)(nil)

func (d *txnDelegator) getInitialState(ctx context.Context, ignoreNotFound bool) (*storage.ObjState, error) {
	obj, exists, err := d.cacher.get(ctx, d.key, "0", true)
	if err != nil {
		klog.Errorf("Watch cache get failed for %s: %v", d.key, err)
		return d.GetCurrentState(ctx, ignoreNotFound)
	}
	if !exists {
		return d.GetCurrentState(ctx, ignoreNotFound)
	}
	// DeepCopy before GetStateFromObject clears and restores ResourceVersion for encoding.
	return storage.GetStateFromObject(d.cacher.versioner, d.cacher.codec, obj.DeepCopyObject(), false)
}

// DeleteOnce implements storage.TxnDelegator.
func (d *txnDelegator) DeleteOnce(ctx context.Context, origState *storage.ObjState, out runtime.Object) (*storage.ObjState, bool, error) {
	err := d.storage.DeleteOnce(ctx, d.key, origState, out)
	if err == nil {
		return origState, true, nil
	}
	if !storage.IsConflict(err) {
		return origState, false, err
	}
	// Ignore failure RV parsing and use empty string to execute consistent read.
	rv, _ := storage.ResourceVersion(err)
	state, err := d.getStateNotOlderThan(ctx, rv, false)
	if err != nil {
		return state, false, err
	}
	if state.Rev == origState.Rev {
		return state, false, storage.NewInternalError(fmt.Errorf("watch cache returned unchanged revision %d after delete conflict on %s", state.Rev, d.key))
	}
	return state, false, nil
}

// UpdateOnce implements storage.TxnDelegator.
func (d *txnDelegator) UpdateOnce(ctx context.Context, origState *storage.ObjState, obj runtime.Object, ttl uint64, destination runtime.Object, ignoreNotFound bool) (*storage.ObjState, bool, error) {
	state, err := d.storage.UpdateOnce(ctx, d.key, origState, obj, ttl, destination, ignoreNotFound)
	if err == nil {
		return state, true, nil
	}
	if !storage.IsConflict(err) {
		return state, false, err
	}
	if state != nil {
		// UpdateOnce refreshed state from storage during a no-op write check.
		return state, false, nil
	}
	// Ignore failure RV parsing and use empty string to execute consistent read.
	rv, _ := storage.ResourceVersion(err)
	state, err = d.getStateNotOlderThan(ctx, rv, ignoreNotFound)
	if err != nil {
		return nil, false, err
	}
	// When origState.Rev == 0 (create-on-update with ignoreNotFound), a concurrent create
	// followed by a delete before the cache is read legitimately yields state.Rev == 0.
	if origState.Rev != 0 && state.Rev == origState.Rev {
		return state, false, storage.NewInternalError(fmt.Errorf("watch cache returned unchanged revision %d after update conflict on %s", state.Rev, d.key))
	}
	return state, false, nil
}

// GetCurrentState implements storage.TxnDelegator.
func (d *txnDelegator) GetCurrentState(ctx context.Context, ignoreNotFound bool) (*storage.ObjState, error) {
	return d.getStateNotOlderThan(ctx, "", ignoreNotFound)
}

func (d *txnDelegator) getStateNotOlderThan(ctx context.Context, resourceVersion string, ignoreNotFound bool) (*storage.ObjState, error) {
	obj, exists, err := d.cacher.get(ctx, d.key, resourceVersion, ignoreNotFound)
	if err != nil {
		return nil, err
	}
	if exists {
		obj = obj.DeepCopyObject()
	} else {
		obj = d.cacher.newFunc()
		if err := runtime.SetZeroValue(obj); err != nil {
			return nil, fmt.Errorf("unable to zero object: %w", err)
		}
	}
	return storage.GetStateFromObject(d.cacher.versioner, d.cacher.codec, obj, true)
}
