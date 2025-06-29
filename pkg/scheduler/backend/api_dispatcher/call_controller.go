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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/utils/buffer"
)

// queuedAPICall contains an API call and other fields needed for its processing by the callController.
type queuedAPICall struct {
	fwk.APICall
	// onFinish is a channel to which the result of the API call's execution is sent in a non-blocking way.
	// sendOnFinish helper should always be used instead of a direct channel.
	onFinish chan<- error
	// timestamp holds the time when the call was added to the APIDispatcher
	// and is used to verify if the call details have been updated while this call was in-flight.
	timestamp time.Time
}

// sendOnFinish performs a non-blocking send of an error to the onFinish channel.
// This method shoud be used instead of direct operation on the channel.
func (qc *queuedAPICall) sendOnFinish(err error) {
	if qc.onFinish != nil {
		select {
		case qc.onFinish <- err:
		default:
		}
	}
}

// callController manages the state and lifecycle of API calls. It handles all
// logic related to reconciliation, queuing, and finalization of calls.
type callController struct {
	lock   sync.RWMutex
	cond   *sync.Cond
	closed bool

	// apiCallRelevances maps all possible APICallTypes to a relevance value.
	// A more relevant API call should overwrite a less relevant one for the same object.
	// Types of the same relevance should only be defined for different object types.
	apiCallRelevances fwk.APICallRelevances

	// apiCalls stores details of API calls for objects that are currently enqueued or in-flight.
	// Must be used under lock.
	apiCalls map[types.UID]*queuedAPICall
	// callsQueue is a FIFO queue that stores the object UIDs of API calls waiting to be executed.
	// Must be used under lock.
	callsQueue buffer.Ring[types.UID]
	// inFlightEntities stores object UIDs of API calls that are currently in-flight
	// (i.e., their execution goroutine has been dispatched).
	// Must be used under lock.
	inFlightEntities sets.Set[types.UID]
}

// newCallController returns a new callController object.
func newCallController(apiCallRelevances fwk.APICallRelevances) *callController {
	q := callController{
		apiCallRelevances: apiCallRelevances,
		apiCalls:          make(map[types.UID]*queuedAPICall),
		inFlightEntities:  sets.New[types.UID](),
	}
	q.cond = sync.NewCond(&q.lock)

	return &q
}

// isLessRelevant returns true if newCall is less relevant than oldCall.
func (cc *callController) isLessRelevant(oldCall, newCall fwk.APICallType) bool {
	return cc.apiCallRelevances[newCall] < cc.apiCallRelevances[oldCall]
}

// isEquallyRelevant returns true if newCall is equally relevant to oldCall.
func (cc *callController) isEquallyRelevant(oldCall, newCall fwk.APICallType) bool {
	return cc.apiCallRelevances[newCall] == cc.apiCallRelevances[oldCall]
}

// reconcile compares a new API call with an existing one for the same object.
// Based on their relevance and type, it determines whether to skip the new call,
// overwrite the old one, or merge them.
// Must be called under cc.lock.
func (cc *callController) reconcile(oldAPICall, apiCall *queuedAPICall) error {
	objectUID := apiCall.UID()

	if cc.isLessRelevant(oldAPICall.CallType(), apiCall.CallType()) {
		// The new API call is less relevant than the existing one, so skip it.
		err := fmt.Errorf("a more relevant call is already enqueued for this object: %w", fwk.ErrCallSkipped)
		apiCall.sendOnFinish(err)
		return err
	}
	if oldAPICall.CallType() != apiCall.CallType() {
		if cc.isEquallyRelevant(oldAPICall.CallType(), apiCall.CallType()) {
			// If call types are different, but are equally relevant, the relevance is misconfigured.
			return fmt.Errorf("relevance misconfiguration: call types %q and %q have the same relevance, but must be different for the same object",
				oldAPICall.CallType(), apiCall.CallType())
		}
		// API call types don't match, so we overwrite the old one.
		oldAPICall.sendOnFinish(fmt.Errorf("a more relevant call was enqueued for this object: %w", fwk.ErrCallOverwritten))
		return nil
	}

	// Merge API calls if they are of the same type for the same object.
	needsCall, err := apiCall.Merge(oldAPICall)
	if err != nil {
		return fmt.Errorf("failed to merge API calls: %w", err)
	}
	if oldAPICall.onFinish != nil {
		oldAPICall.sendOnFinish(fmt.Errorf("a call of the same type was enqueued for this object: %w", fwk.ErrCallOverwritten))
	}
	if !needsCall && !cc.inFlightEntities.Has(objectUID) {
		// The call can be skipped, as the merge resulted in a no-op and the original call is not in-flight.
		cc.removePending(apiCall)
		return fmt.Errorf("call does not need to be executed because after merging it has no effect: %w", fwk.ErrCallSkipped)
	}

	// If we still need the API call or the previous one is in-flight, we proceed.
	return nil
}

// enqueue updates the controller's state with the given apiCall.
// It always replaces the existing entry in the apiCalls map.
// If no previous call was present for the object (oldCallPresent is false),
// it adds the object's UID to the processing queue.
// This prevents concurrent execution for the same object.
// Must be called under cc.lock.
func (cc *callController) enqueue(apiCall *queuedAPICall, oldCallPresent bool) {
	objectUID := apiCall.UID()
	cc.apiCalls[objectUID] = apiCall
	if oldCallPresent {
		// If another API call for this object is already present (i.e., is pending or in-flight),
		// don't add this new call to the queue. The new call will be processed
		// after the currently in-flight call is finalized.
		return
	}
	cc.callsQueue.WriteOne(objectUID)
	cc.cond.Broadcast()
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

// removePending removes a pending API call from the queue and its associated data.
// It notifies the caller that the call was skipped, as it was determined to be a no-op.
// This function is intended to be used on calls that have not yet been popped for execution
// (i.e., are not in-flight).
// Must be called under cc.lock.
func (cc *callController) removePending(apiCall *queuedAPICall) {
	objectUID := apiCall.UID()
	delete(cc.apiCalls, objectUID)
	if !cc.inFlightEntities.Has(objectUID) {
		cc.callsQueue = *removeFromQueue(&cc.callsQueue, objectUID)
	}
	apiCall.sendOnFinish(fmt.Errorf("call does not need to be executed because after updating an object it has no effect: %w", fwk.ErrCallSkipped))
}

// add adds a new apiCall to the queue.
// If an API call for the same object is already present in the queue,
// it tries to skip, overwrite, or merge the calls based on their relevance and type.
func (cc *callController) add(apiCall *queuedAPICall) error {
	cc.lock.Lock()
	defer cc.lock.Unlock()

	oldAPICall, ok := cc.apiCalls[apiCall.UID()]
	if ok {
		err := cc.reconcile(oldAPICall, apiCall)
		if err != nil {
			return err
		}
	}
	cc.enqueue(apiCall, ok)

	return nil
}

// pop pops the first object UID from the queue and returns the corresponding API call details.
func (cc *callController) pop() (*queuedAPICall, error) {
	cc.lock.Lock()
	defer cc.lock.Unlock()

	for cc.callsQueue.Len() == 0 {
		if cc.closed {
			return nil, nil
		}
		// Wait for an API call to become available.
		cc.cond.Wait()
	}

	objectUID, ok := cc.callsQueue.ReadOne()
	if !ok {
		return nil, errors.New("api calls queue is empty")
	}
	apiCall, ok := cc.apiCalls[objectUID]
	if !ok {
		return nil, fmt.Errorf("object %s is not present in a map with API calls details", objectUID)
	}
	cc.inFlightEntities.Insert(objectUID)

	return apiCall, nil
}

// finalize handles a completed API call.
// If a new call for the same object arrived while the original was in-flight, it re-queues the object for processing.
// Otherwise, it removes the completed call's details from the queue.
// This method must be called after a call's execution is finished.
func (cc *callController) finalize(apiCall *queuedAPICall) {
	cc.lock.Lock()
	defer cc.lock.Unlock()

	objectUID := apiCall.UID()
	newAPICall := cc.apiCalls[objectUID]
	if newAPICall.timestamp.Equal(apiCall.timestamp) {
		// The API call in the map hasn't changed, so we can remove it.
		delete(cc.apiCalls, objectUID)
	} else {
		// The API call in the map has changed, so re-queue the object for the new call to be processed.
		cc.callsQueue.WriteOne(objectUID)
		cc.cond.Broadcast()
	}
	cc.inFlightEntities.Delete(objectUID)
}

// updateObject updates the given object using the details of a pending API call for that object,
// if one exists in the queue, and returns the updated object.
func (cc *callController) updateObject(obj metav1.Object) (metav1.Object, error) {
	cc.lock.Lock()
	defer cc.lock.Unlock()

	objectUID := obj.GetUID()
	apiCall, ok := cc.apiCalls[objectUID]
	if !ok {
		return obj, nil
	}

	needsCall, updatedObj, err := apiCall.Update(obj)
	if !needsCall && !cc.inFlightEntities.Has(objectUID) {
		// The call can be removed, as the update resulted in a no-op and the call is not in-flight.
		cc.removePending(apiCall)
	}

	return updatedObj, err
}

// close shuts down the callController.
func (cc *callController) close() {
	cc.lock.Lock()
	defer cc.lock.Unlock()

	cc.closed = true
	cc.cond.Broadcast()
}
