/*
Copyright 2018 The Kubernetes Authors.

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

package endpoint

import (
	"sync"
	"time"

	"k8s.io/klog"
)

/*
TriggerTimeTracker is a util used to compute the EndpointsLastChangeTriggerTime annotation.
See the documentation of the EndpointsLastChangeTriggerTime annotation for more details about why
this annotation and class is needed.

This class was designed to work with the EndpointsController. It's based on the following
assumptions.
1. Client of this util performs two operations, it observes events and runs the sync function.
2. Multiple events can be observed per single sync function run (there is batching).
3. Runs of sync function are mutually exclusive per endpoints object, i.e. for a single endpoints
   object there can be no two sync functions running at the same time.
4. Observe function and sync function can run in parallel.
5. Events are observed in the chronological order, i.e. if T(E1) < T(E2) then E1 will be
   observed before E2.
6. Sync function doesn't use state from the observed events directly. It uses the "list" method
   that returns objects in a state that is AT LEAST as fresh as what was observed in the events,
   i.e. it's possible that the list method will return a state that wasn't yet observed, but
   never the other way around.

Please note that the EndpointsController satisfied all this assumptions in a very strict way.

// TODO(mmat@google.com): Document the interleaving that makes this class to return wrong time.
// Refer to scalability tests results to make it clear how probable is that.
*/
type triggerTimeTracker struct {
	// All maps below are indexed by the Endpoints object namespaced name.

	// lastSyncMinTriggerTime is a map storing the min trigger change time that was saved when the
	// endpoints object was last changed. This map will be updated only in the updateAndReset method.
	lastSyncMinTriggerTime map[string]time.Time

	// lastSyncMaxTriggerTime is a map similar to the lastMinTriggerTime but stores the last *max*
	// trigger change time.
	lastSyncMaxTriggerTime map[string]time.Time

	// minTriggerTime  is a map storing min trigger change time observed since the last updated of the
	// endpoints object. In contrast to lastSavedMinTriggerTime and lastSavedMaxTriggerTime this map
	// will be updated in the observe method, i.e. every time a trigger change is observed
	// (e.g. Pod added / updated / removed) and reset in the updateAndReset method.
	minTriggerTime map[string]time.Time

	// isListingObjects is a map from the endpoints object namespaced name to the bool value that
	// reflects whether the client is currently listing the objects via the list function.
	isListingObjects map[string]bool

	// dirtyTriggerTimes is a map storing all trigger times that were observed during the time the
	// objects were listed in the "list" function. We need this multi-map to make sure that we haven't
	// missed any event due a unfortunate race condition.
	dirtyTriggerTimes map[string][]time.Time

	// mutex guarding this util.
	mutex sync.Mutex
}

// Creates new instance of the TriggerTimeTracker.
func newTriggerTimeTracker() *triggerTimeTracker {
	return &triggerTimeTracker{
		lastSyncMaxTriggerTime : make(map[string]time.Time),
		lastSyncMinTriggerTime : make(map[string]time.Time),
		minTriggerTime : make(map[string]time.Time),
		dirtyTriggerTimes : make(map[string][]time.Time),
		isListingObjects : make(map[string]bool),
	}
}

// Method, updating the minTriggerTime, that should be called every time an event is observed.
func (this *triggerTimeTracker) observe(key string, triggerTime time.Time) {
	this.mutex.Lock()
	defer this.mutex.Unlock()

	if this.isListingObjects[key] {
		this.dirtyTriggerTimes[key] = append(this.dirtyTriggerTimes[key], triggerTime)
		return
	}

	if !triggerTime.After(this.lastSyncMaxTriggerTime[key]) {
		// Trigger was already processed
		if triggerTime.Before(this.lastSyncMinTriggerTime[key]) {
			// Oops, we exported a wrong time in the last processing. Increment the error counter.
 			LastChangeTriggerTimeMiscalculated.Inc()
			klog.Warningf("Miscalculated LastChangeTriggerTime annotation for service %s. " +
				"Should export: %s, exported %s", key, triggerTime, this.lastSyncMinTriggerTime[key])

			// Correct the lastSyncMinTriggerTime so we don't increment the error again for the same
			// batch. It could happen if the trigger change times were T0, T1, T2 and we exported T2 in
			// the last sync. Later we would observe the T0 and T1 trigger change times and because both
			// T0<T2 and T1 < T2 we would export the metric twice.
			this.lastSyncMinTriggerTime[key] = triggerTime
		}
		return
	}

	// If we are here it means that the triggerTime is after lastSyncMaxTriggerTime. If this is first
	// such triggerTime per batch, i.e. minTriggerTime is not set, then we should set it. If it was
	// already set then we don't need to set it anymore for this batch because events are in the
	// chronological order.
	if _, ok := this.minTriggerTime[key]; !ok {
		this.minTriggerTime[key] = triggerTime
	}
}

// Method to be called directly before the call to the "list" function.
func (this *triggerTimeTracker) startListing(key string) {
	this.mutex.Lock()
	defer this.mutex.Unlock()
	this.isListingObjects[key] = true
}

// Method that should be called directly after the listing objects has ended. It will reset the
// state for the given endpoints key and return the time that should be exported as the
// EndpointsLastChangeTriggerTime annotation. It may happen that the method will return nil, in such
// case the annotation shouldn't be exported.
func (this *triggerTimeTracker) stopListingAndReset(
		key string, triggerTimesFromListing []time.Time) (*time.Time){
	this.mutex.Lock()
	defer this.mutex.Unlock()

	// Clear dirty times and isListing state.
	defer func() {
		this.isListingObjects[key] = false
		this.dirtyTriggerTimes[key] = nil
	}()

	var endpointsLastChangeTriggerTime time.Time

	if _, ok := this.minTriggerTime[key]; !ok {
		// There was no event observed that set the minTriggerTime.

		minTriggerTime := func() *time.Time {
			// Try in the dirty times.
			if p := minGreaterThan(this.dirtyTriggerTimes[key], this.lastSyncMaxTriggerTime[key]);
					p != nil {
				return p
			}
			// If nothing found in the dirty times, try in the times from listing objects.
			if p := minGreaterThan(triggerTimesFromListing, this.lastSyncMaxTriggerTime[key]); p != nil {
				return p
			}

			return nil
		}()

		if minTriggerTime == nil {
			// Nothing was observed, nothing in the dirty set, nothing fresh enough in listing times.
			// Return nil, there is nothing to export this time.
			return nil
		}

		this.minTriggerTime[key] = *minTriggerTime
	}

	endpointsLastChangeTriggerTime = this.minTriggerTime[key]
	delete(this.minTriggerTime, key)

	maxTimeFromListing := max(triggerTimesFromListing)
	// See if there is anything in dirty times that could have been a potential new minTriggerTime.
	if p := minGreaterThan(this.dirtyTriggerTimes[key], maxTimeFromListing); p != nil {
		this.minTriggerTime[key] = *p
	}

	// Updated the lastSync min and max.
	this.lastSyncMinTriggerTime[key] = endpointsLastChangeTriggerTime
	this.lastSyncMaxTriggerTime[key] = maxTimeFromListing
	return &endpointsLastChangeTriggerTime
}


// ----- Util Functions -----

func min(times []time.Time) (minTime time.Time) {
	for _, t := range times {
		if minTime.IsZero() || t.Before(minTime) {
			minTime = t
		}
	}
	return minTime
}

func max(times []time.Time) (maxTime time.Time) {
	for _, t := range times {
		if t.After(maxTime) {
			maxTime = t
		}
	}
	return maxTime
}

func minGreaterThan(times []time.Time, greaterThan time.Time) (minTime *time.Time) {
	for i, t := range times {
		if t.After(greaterThan) && (minTime == nil || t.Before(*minTime)) {
			minTime = &times[i] // Cannot use &t as t is a variable that is updated in every iteration.
		}
	}
	return minTime
}
