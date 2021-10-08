/*
Copyright 2020 The Kubernetes Authors.

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

package job

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// uidSetKeyFunc to parse out the key from a uidSet.
var uidSetKeyFunc = func(obj interface{}) (string, error) {
	if u, ok := obj.(*uidSet); ok {
		return u.key, nil
	}
	return "", fmt.Errorf("could not find key for obj %#v", obj)
}

// uidSet holds a key and a set of UIDs. Used by the
// uidTrackingExpectations to remember which UID it has seen/still waiting for.
type uidSet struct {
	sync.RWMutex
	set sets.String
	key string
}

// uidTrackingExpectations tracks the UIDs of Pods the controller is waiting to
// observe tracking finalizer deletions.
type uidTrackingExpectations struct {
	store cache.Store
}

// GetUIDs is a convenience method to avoid exposing the set of expected uids.
// The returned set is not thread safe, all modifications must be made holding
// the uidStoreLock.
func (u *uidTrackingExpectations) getSet(controllerKey string) *uidSet {
	if obj, exists, err := u.store.GetByKey(controllerKey); err == nil && exists {
		return obj.(*uidSet)
	}
	return nil
}

func (u *uidTrackingExpectations) getExpectedUIDs(controllerKey string) sets.String {
	uids := u.getSet(controllerKey)
	if uids == nil {
		return nil
	}
	uids.RLock()
	set := sets.NewString(uids.set.UnsortedList()...)
	uids.RUnlock()
	return set
}

// ExpectDeletions records expectations for the given deleteKeys, against the
// given job-key.
// This is thread-safe across different job keys.
func (u *uidTrackingExpectations) expectFinalizersRemoved(jobKey string, deletedKeys []string) error {
	klog.V(4).InfoS("Expecting tracking finalizers removed", "job", jobKey, "podUIDs", deletedKeys)

	uids := u.getSet(jobKey)
	if uids == nil {
		uids = &uidSet{
			key: jobKey,
			set: sets.NewString(),
		}
		if err := u.store.Add(uids); err != nil {
			return err
		}
	}
	uids.Lock()
	uids.set.Insert(deletedKeys...)
	uids.Unlock()
	return nil
}

// FinalizerRemovalObserved records the given deleteKey as a deletion, for the given job.
func (u *uidTrackingExpectations) finalizerRemovalObserved(jobKey, deleteKey string) {
	uids := u.getSet(jobKey)
	if uids != nil {
		uids.Lock()
		if uids.set.Has(deleteKey) {
			klog.V(4).InfoS("Observed tracking finalizer removed", "job", jobKey, "podUID", deleteKey)
			uids.set.Delete(deleteKey)
		}
		uids.Unlock()
	}
}

// DeleteExpectations deletes the UID set.
func (u *uidTrackingExpectations) deleteExpectations(jobKey string) {
	if err := u.store.Delete(jobKey); err != nil {
		klog.ErrorS(err, "deleting tracking annotation UID expectations", "job", jobKey)
	}
}

// NewUIDTrackingControllerExpectations returns a wrapper around
// ControllerExpectations that is aware of deleteKeys.
func newUIDTrackingExpectations() *uidTrackingExpectations {
	return &uidTrackingExpectations{store: cache.NewStore(uidSetKeyFunc)}
}
