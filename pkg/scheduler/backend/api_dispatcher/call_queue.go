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

package apidispatcher

import (
	"errors"
	"fmt"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/utils/buffer"
)

// queuedAPICall contains an API call and other fields needed for its processing by the callQueue.
type queuedAPICall struct {
	fwk.APICall
	// onFinish is a channel to which the result of the API call's execution is sent in a non-blocking way.
	// sendOnFinish helper should always be used instead of a direct channel.
	onFinish chan<- error
	// callID provides a unique identity for this enqueued call.
	// It is used during finalization to differentiate this instance from a newer one that may have replaced it
	// while this call was in-flight.
	callID int
}

// sendOnFinish performs a non-blocking send of an error to the onFinish channel.
// This method should be used instead of direct operation on the channel.
func (qc *queuedAPICall) sendOnFinish(err error) {
	if qc.onFinish != nil {
		select {
		case qc.onFinish <- err:
		default:
		}
	}
}

// callQueue manages the state and lifecycle of API calls.
// It handles all logic related to merging, queuing, and finalization of calls.
//
// For any given object (identified by its UID), at most one API
// call is present in the callsQueue and apiCalls at any time. This prevents race conditions
// and ensures sequential execution for operations on the same object.
//
// When a new call is added for an object that is already being tracked,
// the controller reconciles them by merging the calls, overwriting the older one,
// or skipping the new one entirely based on relevance.
//
// When an existing call is in-progress and a new one for the same object is added,
// it will be inserted to the callsQueue when the existing one is finished (finalized).
//
// The controller tracks the state of all calls (pending, in-flight)
// to coordinate the asynchronous execution flow managed by the APIDispatcher.
type callQueue struct {
	lock   sync.RWMutex
	cond   *sync.Cond
	closed bool

	// apiCallRelevances maps all possible APICallTypes to a relevance value.
	// A more relevant API call should overwrite a less relevant one for the same object.
	// Types of the same relevance should only be defined for different object types.
	apiCallRelevances fwk.APICallRelevances
	// callIDCounter is a monotonically increasing counter used to assign a unique callID
	// to each queuedAPICall added to the callQueue.
	callIDCounter int

	// apiCalls stores details of API calls for objects that are currently enqueued or in-flight.
	// Must be used under lock.
	apiCalls map[types.UID]*queuedAPICall
	// callsQueue is a FIFO queue that stores the object UIDs of (pending) API calls waiting to be executed.
	// Must be used under lock.
	callsQueue buffer.Ring[types.UID]
	// inFlightEntities stores object UIDs of API calls that are currently in-flight
	// (i.e., their execution goroutine has been dispatched).
	// Must be used under lock.
	inFlightEntities sets.Set[types.UID]
}

// newCallQueue returns a new callQueue object.
func newCallQueue(apiCallRelevances fwk.APICallRelevances) *callQueue {
	q := callQueue{
		apiCallRelevances: apiCallRelevances,
		apiCalls:          make(map[types.UID]*queuedAPICall),
		inFlightEntities:  sets.New[types.UID](),
	}
	q.cond = sync.NewCond(&q.lock)

	return &q
}

// isLessRelevant returns true if newCall is less relevant than oldCall.
func (cq *callQueue) isLessRelevant(oldCall, newCall fwk.APICallType) bool {
	return cq.apiCallRelevances[newCall] < cq.apiCallRelevances[oldCall]
}

// merge compares a new API call with an existing one for the same object.
// Based on their relevance and type, it determines whether to skip the new call,
// overwrite the old one, or merge them.
// Must be called under cq.lock.
func (cq *callQueue) merge(oldAPICall, apiCall *queuedAPICall) error {
	if cq.isLessRelevant(oldAPICall.CallType(), apiCall.CallType()) {
		// The new API call is less relevant than the existing one, so skip it.
		err := fmt.Errorf("a more relevant call is already enqueued for this object: %w", fwk.ErrCallSkipped)
		apiCall.sendOnFinish(err)
		return err
	}
	if oldAPICall.CallType() != apiCall.CallType() {
		// API call types don't match, so we overwrite the old one.
		// Update the pending calls metric if the old call is not in-flight.
		if !cq.inFlightEntities.Has(oldAPICall.UID()) {
			metrics.AsyncAPIPendingCalls.WithLabelValues(string(oldAPICall.CallType())).Dec()
			metrics.AsyncAPIPendingCalls.WithLabelValues(string(apiCall.CallType())).Inc()
		}
		oldAPICall.sendOnFinish(fmt.Errorf("a more relevant call was enqueued for this object: %w", fwk.ErrCallOverwritten))
		return nil
	}

	// Send a concrete APICall object to allow for a type assertion.
	err := apiCall.Merge(oldAPICall.APICall)
	if err != nil {
		err := fmt.Errorf("failed to merge API calls: %w", err)
		apiCall.sendOnFinish(err)
		return err
	}
	oldAPICall.sendOnFinish(fmt.Errorf("a call of the same type was enqueued for this object: %w", fwk.ErrCallOverwritten))

	// If we still need the API call or the previous one is in-flight, we proceed.
	return nil
}

// enqueue handles the logic of adding a new or updated API call to the controller's state.
//
// First, it checks if the apiCall is a no-op. If it is, and no other call for the same object
// is currently in-flight, the call is skipped and an ErrCallSkipped is returned.
//
// If the call is not skipped, it is stored internally. If no previous call for the object
// was being tracked (oldCallPresent is false), the object's UID is added to the processing
// queue to be picked up by a worker.
//
// This logic ensures only one call per object is active in the queue at a time.
// Must be called under cq.lock.
func (cq *callQueue) enqueue(apiCall *queuedAPICall, oldCallPresent bool) error {
	noOp := apiCall.IsNoOp()
	if noOp && !cq.inFlightEntities.Has(apiCall.UID()) {
		// The call can be skipped, as it is a no-op and the old call is not in-flight.
		if oldCallPresent {
			cq.removePending(apiCall.UID())
		}
		apiCall.sendOnFinish(fmt.Errorf("call does not need to be executed because it has no effect: %w", fwk.ErrCallSkipped))
		return fmt.Errorf("call does not need to be executed because it has no effect: %w", fwk.ErrCallSkipped)
	}

	objectUID := apiCall.UID()
	cq.apiCalls[objectUID] = apiCall
	if oldCallPresent {
		// If another API call for this object is already present (i.e., is pending or in-flight),
		// don't add this new call to the queue. The new call will be processed
		// after the currently in-flight call is finalized.
		return nil
	}
	cq.callsQueue.WriteOne(objectUID)
	metrics.AsyncAPIPendingCalls.WithLabelValues(string(apiCall.CallType())).Inc()
	cq.cond.Broadcast()
	return nil
}

// removeFromQueue removes the objectUID from the queue and returns the recreated queue.
func removeFromQueue(queue *buffer.Ring[types.UID], objectUID types.UID) *buffer.Ring[types.UID] {
	newQueue := buffer.NewRing[types.UID](buffer.RingOptions{
		InitialSize: queue.Len(),
		NormalSize:  queue.Cap(),
	})
	for {
		uid, ok := queue.ReadOne()
		if !ok {
			break
		}
		if uid != objectUID {
			newQueue.WriteOne(uid)
		}
	}
	return newQueue
}

// removePending removes a pending API call for the given objectUID from the queue and its associated data.
// This function is intended to be used on calls that have not yet been popped for execution
// (i.e., are not in-flight).
// Must be called under cq.lock.
func (cq *callQueue) removePending(objectUID types.UID) {
	apiCall, ok := cq.apiCalls[objectUID]
	if !ok {
		return
	}
	delete(cq.apiCalls, objectUID)
	if !cq.inFlightEntities.Has(objectUID) {
		cq.callsQueue = *removeFromQueue(&cq.callsQueue, objectUID)
		callType := string(apiCall.CallType())
		metrics.AsyncAPIPendingCalls.WithLabelValues(callType).Dec()
	}
}

// add adds a new apiCall to the queue.
// If an API call for the same object is already present in the queue,
// it tries to skip, overwrite, or merge the calls based on their relevance and type.
func (cq *callQueue) add(apiCall *queuedAPICall) error {
	cq.lock.Lock()
	defer cq.lock.Unlock()

	apiCall.callID = cq.callIDCounter
	cq.callIDCounter++

	oldAPICall, ok := cq.apiCalls[apiCall.UID()]
	if ok {
		err := cq.merge(oldAPICall, apiCall)
		if err != nil {
			return err
		}
	}
	return cq.enqueue(apiCall, ok)
}

// pop pops the first object UID from the queue and returns the corresponding API call details.
// Note: The caller must always call finalize() for the poped apiCall, otherwise it would result in the memory leak.
func (cq *callQueue) pop() (*queuedAPICall, error) {
	cq.lock.Lock()
	defer cq.lock.Unlock()

	for cq.callsQueue.Len() == 0 {
		if cq.closed {
			return nil, nil
		}
		// Wait for an API call to become available.
		cq.cond.Wait()
	}

	objectUID, ok := cq.callsQueue.ReadOne()
	if !ok {
		return nil, errors.New("api calls queue is empty")
	}
	apiCall, ok := cq.apiCalls[objectUID]
	if !ok {
		return nil, fmt.Errorf("object %s is not present in a map with API calls details", objectUID)
	}
	cq.inFlightEntities.Insert(objectUID)
	callType := string(apiCall.CallType())
	metrics.AsyncAPIPendingCalls.WithLabelValues(callType).Dec()

	return apiCall, nil
}

// finalize handles a completed API call.
// If a new call for the same object arrived while the original was in-flight, it re-queues the object for processing.
// Otherwise, it removes the completed call's details from the queue.
// This method must be called after a call's execution is finished.
func (cq *callQueue) finalize(apiCall *queuedAPICall) {
	cq.lock.Lock()
	defer cq.lock.Unlock()

	objectUID := apiCall.UID()
	newAPICall := cq.apiCalls[objectUID]
	if newAPICall.callID == apiCall.callID {
		// The API call in the map hasn't changed, so we can remove it.
		delete(cq.apiCalls, objectUID)
	} else {
		// The API call in the map has changed, so re-queue the object for the new call to be processed.
		cq.callsQueue.WriteOne(objectUID)
		callType := string(newAPICall.CallType())
		metrics.AsyncAPIPendingCalls.WithLabelValues(callType).Inc()
		cq.cond.Broadcast()
	}
	cq.inFlightEntities.Delete(objectUID)
}

// syncObject performs a two-way synchronization between the given object
// and a pending API call for that object, if one exists in the queue, and returns the synced object.
func (cq *callQueue) syncObject(obj metav1.Object) (metav1.Object, error) {
	cq.lock.Lock()
	defer cq.lock.Unlock()

	objectUID := obj.GetUID()
	apiCall, ok := cq.apiCalls[objectUID]
	if !ok {
		return obj, nil
	}

	syncedObj, err := apiCall.Sync(obj)
	noOp := apiCall.IsNoOp()
	if noOp && !cq.inFlightEntities.Has(apiCall.UID()) {
		// The call can be removed, as the sync resulted in a no-op and the call is not in-flight.
		cq.removePending(apiCall.UID())
		apiCall.sendOnFinish(fmt.Errorf("call does not need to be executed because after sync it has no effect: %w", fwk.ErrCallSkipped))
	}

	return syncedObj, err
}

// close shuts down the callQueue.
func (cq *callQueue) close() {
	cq.lock.Lock()
	defer cq.lock.Unlock()

	cq.closed = true
	cq.cond.Broadcast()
}
