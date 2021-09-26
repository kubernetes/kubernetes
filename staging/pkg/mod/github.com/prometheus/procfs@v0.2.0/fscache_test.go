// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"reflect"
	"testing"
)

func TestFscacheinfo(t *testing.T) {
	expected := Fscacheinfo{
		IndexCookiesAllocated:                              3,
		DataStorageCookiesAllocated:                        67877,
		SpecialCookiesAllocated:                            0,
		ObjectsAllocated:                                   67473,
		ObjectAllocationsFailure:                           0,
		ObjectsAvailable:                                   67473,
		ObjectsDead:                                        388,
		ObjectsWithoutCoherencyCheck:                       12,
		ObjectsWithCoherencyCheck:                          33,
		ObjectsNeedCoherencyCheckUpdate:                    44,
		ObjectsDeclaredObsolete:                            55,
		PagesMarkedAsBeingCached:                           547164,
		UncachePagesRequestSeen:                            364577,
		AcquireCookiesRequestSeen:                          67880,
		AcquireRequestsWithNullParent:                      98,
		AcquireRequestsRejectedNoCacheAvailable:            25,
		AcquireRequestsSucceeded:                           67780,
		AcquireRequestsRejectedDueToError:                  39,
		AcquireRequestsFailedDueToEnomem:                   26,
		LookupsNumber:                                      67473,
		LookupsNegative:                                    67470,
		LookupsPositive:                                    58,
		ObjectsCreatedByLookup:                             67473,
		LookupsTimedOutAndRequed:                           85,
		InvalidationsNumber:                                14,
		InvalidationsRunning:                               13,
		UpdateCookieRequestSeen:                            7,
		UpdateRequestsWithNullParent:                       3,
		UpdateRequestsRunning:                              8,
		RelinquishCookiesRequestSeen:                       394,
		RelinquishCookiesWithNullParent:                    1,
		RelinquishRequestsWaitingCompleteCreation:          2,
		RelinquishRetries:                                  3,
		AttributeChangedRequestsSeen:                       6,
		AttributeChangedRequestsQueued:                     5,
		AttributeChangedRejectDueToEnobufs:                 4,
		AttributeChangedFailedDueToEnomem:                  3,
		AttributeChangedOps:                                2,
		AllocationRequestsSeen:                             20,
		AllocationOkRequests:                               19,
		AllocationWaitingOnLookup:                          18,
		AllocationsRejectedDueToEnobufs:                    17,
		AllocationsAbortedDueToErestartsys:                 16,
		AllocationOperationsSubmitted:                      15,
		AllocationsWaitedForCPU:                            14,
		AllocationsAbortedDueToObjectDeath:                 13,
		RetrievalsReadRequests:                             151959,
		RetrievalsOk:                                       82823,
		RetrievalsWaitingLookupCompletion:                  23467,
		RetrievalsReturnedEnodata:                          69136,
		RetrievalsRejectedDueToEnobufs:                     15,
		RetrievalsAbortedDueToErestartsys:                  69,
		RetrievalsFailedDueToEnomem:                        43,
		RetrievalsRequests:                                 151959,
		RetrievalsWaitingCPU:                               42747,
		RetrievalsAbortedDueToObjectDeath:                  44,
		StoreWriteRequests:                                 225565,
		StoreSuccessfulRequests:                            225565,
		StoreRequestsOnPendingStorage:                      12,
		StoreRequestsRejectedDueToEnobufs:                  13,
		StoreRequestsFailedDueToEnomem:                     14,
		StoreRequestsSubmitted:                             69156,
		StoreRequestsRunning:                               294721,
		StorePagesWithRequestsProcessing:                   225565,
		StoreRequestsDeleted:                               225565,
		StoreRequestsOverStoreLimit:                        43,
		ReleaseRequestsAgainstPagesWithNoPendingStorage:    364512,
		ReleaseRequestsAgainstPagesStoredByTimeLockGranted: 2,
		ReleaseRequestsIgnoredDueToInProgressStore:         43,
		PageStoresCancelledByReleaseRequests:               12,
		VmscanWaiting:                                      66,
		OpsPending:                                         42753,
		OpsRunning:                                         221129,
		OpsEnqueued:                                        628798,
		OpsCancelled:                                       11,
		OpsRejected:                                        88,
		OpsInitialised:                                     377538,
		OpsDeferred:                                        27,
		OpsReleased:                                        377538,
		OpsGarbageCollected:                                37,
		CacheopAllocationsinProgress:                       1,
		CacheopLookupObjectInProgress:                      2,
		CacheopLookupCompleteInPorgress:                    3,
		CacheopGrabObjectInProgress:                        4,
		CacheopInvalidations:                               5,
		CacheopUpdateObjectInProgress:                      6,
		CacheopDropObjectInProgress:                        7,
		CacheopPutObjectInProgress:                         8,
		CacheopAttributeChangeInProgress:                   9,
		CacheopSyncCacheInProgress:                         10,
		CacheopReadOrAllocPageInProgress:                   11,
		CacheopReadOrAllocPagesInProgress:                  12,
		CacheopAllocatePageInProgress:                      13,
		CacheopAllocatePagesInProgress:                     14,
		CacheopWritePagesInProgress:                        15,
		CacheopUncachePagesInProgress:                      16,
		CacheopDissociatePagesInProgress:                   17,
		CacheevLookupsAndCreationsRejectedLackSpace:        18,
		CacheevStaleObjectsDeleted:                         19,
		CacheevRetiredWhenReliquished:                      20,
		CacheevObjectsCulled:                               21,
	}

	have, err := getProcFixtures(t).Fscacheinfo()
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(have, expected) {
		t.Logf("have: %+v", have)
		t.Logf("expected: %+v", expected)
		t.Errorf("structs are not equal")
	}
}
