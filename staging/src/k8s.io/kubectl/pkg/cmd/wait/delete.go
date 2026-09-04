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

package wait

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubectl/pkg/util/interrupt"
)

// IsDeleted is a condition func for waiting for something to be deleted
func IsDeleted(ctx context.Context, info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	if len(info.Name) == 0 {
		return info.Object, false, fmt.Errorf("resource name must be provided")
	}

	gottenObj, initObjGetErr := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Get(ctx, info.Name, metav1.GetOptions{})
	if apierrors.IsNotFound(initObjGetErr) {
		return info.Object, true, nil
	}
	if initObjGetErr != nil {
		// TODO this could do something slightly fancier if we wish
		return info.Object, false, initObjGetErr
	}
	resourceLocation := ResourceLocation{
		GroupResource: info.Mapping.Resource.GroupResource(),
		Namespace:     gottenObj.GetNamespace(),
		Name:          gottenObj.GetName(),
	}
	if uid, ok := o.UIDMap[resourceLocation]; ok {
		if gottenObj.GetUID() != uid {
			return gottenObj, true, nil
		}
	}

	endTime := time.Now().Add(o.Timeout)
	timeout := time.Until(endTime)
	errWaitTimeoutWithName := extendErrWaitTimeout(wait.ErrorInterrupted(nil), info) // nolint:staticcheck // SA1019
	if o.Timeout == 0 {
		// If timeout is zero check if the object exists once only
		if gottenObj == nil {
			return nil, true, nil
		}
		return gottenObj, false, fmt.Errorf("condition not met for %s", info.ObjectName())
	}
	if timeout < 0 {
		// we're out of time
		return info.Object, false, errWaitTimeoutWithName
	}

	intrCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// The reflector backing UntilWithSync treats list and watch failures as
	// retryable, so a caller that may delete an object but may not list or watch
	// it would otherwise retry until the timeout instead of returning. Record the
	// first forbidden error and stop waiting, so the caller can decide what it
	// means. Deletion itself has already succeeded at this point.
	var (
		forbiddenLock sync.Mutex
		forbiddenErr  error
	)
	recordForbidden := func(err error) {
		if !apierrors.IsForbidden(err) {
			return
		}
		forbiddenLock.Lock()
		defer forbiddenLock.Unlock()
		if forbiddenErr == nil {
			forbiddenErr = err
			cancel()
		}
	}

	fieldSelector := fields.OneTermEqualSelector("metadata.name", info.Name).String()
	lw := cache.ToListWatcherWithWatchListSemantics(&cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			list, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).List(ctx, options)
			recordForbidden(err)
			return list, err
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			w, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(ctx, options)
			recordForbidden(err)
			return w, err
		},
	}, o.DynamicClient)

	// this function is used to refresh the cache to prevent timeout waits on resources that have disappeared
	preconditionFunc := func(store cache.Store) (bool, error) {
		obj, exists, err := store.Get(&metav1.ObjectMeta{Namespace: info.Namespace, Name: info.Name})
		if err != nil {
			return true, err
		}
		if !exists {
			// since we're looking for it to disappear we just return here if it no longer exists
			return true, nil
		}
		acc, err := meta.Accessor(obj)
		if err != nil {
			return true, err
		}
		if uid, ok := o.UIDMap[resourceLocation]; ok {
			if acc.GetUID() != uid {
				return true, nil
			}
		}
		return false, nil
	}

	intr := interrupt.New(nil, cancel)
	err := intr.Run(func() error {
		_, err := watchtools.UntilWithSync(intrCtx, lw, &unstructured.Unstructured{}, preconditionFunc, Wait{errOut: o.ErrOut}.IsDeleted)
		if errors.Is(err, context.DeadlineExceeded) {
			return errWaitTimeoutWithName
		}
		return err
	})
	if err != nil {
		// A recorded denial is the reason the wait stopped, so report it rather
		// than the context cancellation it triggered. Only consult it once the
		// wait has failed: if deletion was confirmed first, the confirmation stands.
		forbiddenLock.Lock()
		stoppedByForbidden := forbiddenErr
		forbiddenLock.Unlock()
		if stoppedByForbidden != nil {
			// Cancelling the wait may have pre-empted a precondition that was
			// about to be satisfied, so confirm through the get that already
			// succeeded above before reporting the denial.
			if _, getErr := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Get(ctx, info.Name, metav1.GetOptions{}); apierrors.IsNotFound(getErr) {
				return info.Object, true, nil
			}
			return gottenObj, false, stoppedByForbidden
		}
		if wait.Interrupted(err) { // nolint:staticcheck // SA1019
			return gottenObj, false, errWaitTimeoutWithName
		}
		return gottenObj, false, err
	}

	return gottenObj, true, nil
}

// Wait has helper methods for handling watches, including error handling.
type Wait struct {
	errOut io.Writer
}

// IsDeleted returns true if the object is deleted. It prints any errors it encounters.
func (w Wait) IsDeleted(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Error:
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server if the error is unrecoverable.
		err := apierrors.FromObject(event.Object)
		fmt.Fprintf(w.errOut, "error: An error occurred while waiting for the object to be deleted: %v", err)
		return false, nil
	case watch.Deleted:
		return true, nil
	default:
		return false, nil
	}
}
