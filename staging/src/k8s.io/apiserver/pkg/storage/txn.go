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

package storage

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
)

type TxnDelegator interface {
	GetCurrentState(ctx context.Context, ignoreNotFound bool) (*ObjState, error)
	DeleteOnce(ctx context.Context, origState *ObjState, out runtime.Object) (state *ObjState, success bool, err error)
	UpdateOnce(ctx context.Context, origState *ObjState, obj runtime.Object, ttl uint64, destination runtime.Object, ignoreNotFound bool) (state *ObjState, success bool, err error)
}

type ObjState struct {
	Obj  runtime.Object
	Meta *ResponseMeta
	Rev  int64
	Data []byte
	// Current is true when this state was loaded from storage, and false when
	// it was initialized in-memory from cachedExistingObject.
	Current bool
	// Stale is true when TransformFromStorage reported that the value in storage
	// needs to be rewritten via TransformToStorage even if data has not changed.
	Stale bool
}

func GetStateFromObject(versioner Versioner, codec runtime.Codec, obj runtime.Object, current bool) (*ObjState, error) {
	state := &ObjState{
		Obj:     obj,
		Meta:    &ResponseMeta{},
		Current: current,
	}

	rv, err := versioner.ObjectResourceVersion(obj)
	if err != nil {
		return nil, fmt.Errorf("couldn't get resource version: %v", err)
	}
	state.Rev = int64(rv)
	state.Meta.ResourceVersion = uint64(state.Rev)

	// Compute the serialized form - for that we need to temporarily clean
	// its resource version field (those are not stored in etcd).
	if err := versioner.PrepareObjectForStorage(obj); err != nil {
		return nil, fmt.Errorf("PrepareObjectForStorage failed: %v", err)
	}
	state.Data, err = runtime.Encode(codec, obj)
	if err != nil {
		return nil, err
	}
	if err := versioner.UpdateObject(state.Obj, uint64(rv)); err != nil {
		klog.Errorf("failed to update object version: %v", err)
	}
	return state, nil
}

func ConditionalDelete(ctx context.Context, d TxnDelegator, key string, out runtime.Object, preconditions *Preconditions, validateDeletion ValidateObjectFunc, origState *ObjState) (err error) {
	for {
		if preconditions != nil {
			if err := preconditions.Check(key, origState.Obj); err != nil {
				origState, err = refreshOnError(ctx, origState, err, false, d.GetCurrentState)
				if err != nil {
					return err
				}
				continue
			}
		}
		if err := validateDeletion(ctx, origState.Obj); err != nil {
			origState, err = refreshOnError(ctx, origState, err, false, d.GetCurrentState)
			if err != nil {
				return err
			}
			continue
		}

		var success bool
		origState, success, err = d.DeleteOnce(ctx, origState, out)
		if err != nil {
			return err
		}
		if success {
			return nil
		}
	}
}

func GuaranteedUpdate(ctx context.Context, d TxnDelegator, key string, destination runtime.Object, ignoreNotFound bool, preconditions *Preconditions, tryUpdate UpdateFunc, origState *ObjState) (err error) {
	for {
		if err := preconditions.Check(key, origState.Obj); err != nil {
			origState, err = refreshOnError(ctx, origState, err, ignoreNotFound, d.GetCurrentState)
			if err != nil {
				return err
			}
			continue
		}

		ret, ttl, err := updateState(origState, tryUpdate)
		if err != nil {
			origState, err = refreshOnError(ctx, origState, err, ignoreNotFound, d.GetCurrentState)
			if err != nil {
				return err
			}
			continue
		}

		var success bool
		origState, success, err = d.UpdateOnce(ctx, origState, ret, ttl, destination, ignoreNotFound)
		if err != nil {
			return err
		}
		if success {
			return nil
		}
	}
}

// refreshOnError transitions a non-current state to the current state from storage
// when a precondition, validation, or update error occurs. If the state is already
// current, or if the revision in storage matches the cached revision, it returns the
// original error.
func refreshOnError(ctx context.Context, st *ObjState, err error, ignoreNotFound bool, getCurrentState func(ctx context.Context, ignoreNotFound bool) (*ObjState, error)) (*ObjState, error) {
	// If our data is already up to date, return the error
	if st.Current {
		return st, err
	}

	// It's possible we were working with stale data.
	// Remember the revision of the potentially stale data and the resulting error.
	cachedRev := st.Rev
	cachedErr := err

	currentState, err := getCurrentState(ctx, ignoreNotFound)
	if err != nil {
		return nil, err
	}

	// It turns out our cached data was not stale, return the original error.
	if cachedRev == currentState.Rev {
		return currentState, cachedErr
	}

	// Retry
	return currentState, nil
}

func updateState(st *ObjState, userUpdate UpdateFunc) (runtime.Object, uint64, error) {
	ret, ttlPtr, err := userUpdate(st.Obj, *st.Meta)
	if err != nil {
		return nil, 0, err
	}
	var ttl uint64
	if ttlPtr != nil {
		ttl = *ttlPtr
	}
	return ret, ttl, nil
}
