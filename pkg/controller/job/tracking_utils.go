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

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/job/metrics"
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
	set sets.Set[types.UID]
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

func (u *uidTrackingExpectations) getExpectedUIDs(controllerKey string) sets.Set[types.UID] {
	uids := u.getSet(controllerKey)
	if uids == nil {
		return nil
	}
	uids.RLock()
	set := uids.set.Clone()
	uids.RUnlock()
	return set
}

// ExpectDeletions records expectations for the given deleteKeys, against the
// given job-key.
// This is thread-safe across different job keys.
func (u *uidTrackingExpectations) expectFinalizersRemoved(logger klog.Logger, jobKey string, deletedKeys []types.UID) error {
	logger.V(4).Info("Expecting tracking finalizers removed", "key", jobKey, "podUIDs", deletedKeys)

	uids := u.getSet(jobKey)
	if uids == nil {
		uids = &uidSet{
			key: jobKey,
			set: sets.New[types.UID](),
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
func (u *uidTrackingExpectations) finalizerRemovalObserved(logger klog.Logger, jobKey string, deleteKey types.UID) {
	uids := u.getSet(jobKey)
	if uids != nil {
		uids.Lock()
		if uids.set.Has(deleteKey) {
			logger.V(4).Info("Observed tracking finalizer removed", "key", jobKey, "podUID", deleteKey)
			uids.set.Delete(deleteKey)
		}
		uids.Unlock()
	}
}

// DeleteExpectations deletes the UID set.
func (u *uidTrackingExpectations) deleteExpectations(logger klog.Logger, jobKey string) {
	set := u.getSet(jobKey)
	if set != nil {
		if err := u.store.Delete(set); err != nil {
			logger.Error(err, "Could not delete tracking annotation UID expectations", "key", jobKey)
		}
	}
}

// NewUIDTrackingControllerExpectations returns a wrapper around
// ControllerExpectations that is aware of deleteKeys.
func newUIDTrackingExpectations() *uidTrackingExpectations {
	return &uidTrackingExpectations{store: cache.NewStore(uidSetKeyFunc)}
}

func hasJobTrackingFinalizer(pod *v1.Pod) bool {
	for _, fin := range pod.Finalizers {
		if fin == batch.JobTrackingFinalizer {
			return true
		}
	}
	return false
}

func recordFinishedPodWithTrackingFinalizer(oldPod, newPod *v1.Pod) {
	was := isFinishedPodWithTrackingFinalizer(oldPod)
	is := isFinishedPodWithTrackingFinalizer(newPod)
	if was == is {
		return
	}
	var event = metrics.Delete
	if is {
		event = metrics.Add
	}
	metrics.TerminatedPodsTrackingFinalizerTotal.WithLabelValues(event).Inc()
}

func isFinishedPodWithTrackingFinalizer(pod *v1.Pod) bool {
	if pod == nil {
		return false
	}
	return (pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded) && hasJobTrackingFinalizer(pod)
}
