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
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Fscacheinfo represents fscache statistics.
type Fscacheinfo struct {
	// Number of index cookies allocated
	IndexCookiesAllocated uint64
	// data storage cookies allocated
	DataStorageCookiesAllocated uint64
	// Number of special cookies allocated
	SpecialCookiesAllocated uint64
	// Number of objects allocated
	ObjectsAllocated uint64
	// Number of object allocation failures
	ObjectAllocationsFailure uint64
	// Number of objects that reached the available state
	ObjectsAvailable uint64
	// Number of objects that reached the dead state
	ObjectsDead uint64
	// Number of objects that didn't have a coherency check
	ObjectsWithoutCoherencyCheck uint64
	// Number of objects that passed a coherency check
	ObjectsWithCoherencyCheck uint64
	// Number of objects that needed a coherency data update
	ObjectsNeedCoherencyCheckUpdate uint64
	// Number of objects that were declared obsolete
	ObjectsDeclaredObsolete uint64
	// Number of pages marked as being cached
	PagesMarkedAsBeingCached uint64
	// Number of uncache page requests seen
	UncachePagesRequestSeen uint64
	// Number of acquire cookie requests seen
	AcquireCookiesRequestSeen uint64
	// Number of acq reqs given a NULL parent
	AcquireRequestsWithNullParent uint64
	// Number of acq reqs rejected due to no cache available
	AcquireRequestsRejectedNoCacheAvailable uint64
	// Number of acq reqs succeeded
	AcquireRequestsSucceeded uint64
	// Number of acq reqs rejected due to error
	AcquireRequestsRejectedDueToError uint64
	// Number of acq reqs failed on ENOMEM
	AcquireRequestsFailedDueToEnomem uint64
	// Number of lookup calls made on cache backends
	LookupsNumber uint64
	// Number of negative lookups made
	LookupsNegative uint64
	// Number of positive lookups made
	LookupsPositive uint64
	// Number of objects created by lookup
	ObjectsCreatedByLookup uint64
	// Number of lookups timed out and requeued
	LookupsTimedOutAndRequed uint64
	InvalidationsNumber      uint64
	InvalidationsRunning     uint64
	// Number of update cookie requests seen
	UpdateCookieRequestSeen uint64
	// Number of upd reqs given a NULL parent
	UpdateRequestsWithNullParent uint64
	// Number of upd reqs granted CPU time
	UpdateRequestsRunning uint64
	// Number of relinquish cookie requests seen
	RelinquishCookiesRequestSeen uint64
	// Number of rlq reqs given a NULL parent
	RelinquishCookiesWithNullParent uint64
	// Number of rlq reqs waited on completion of creation
	RelinquishRequestsWaitingCompleteCreation uint64
	// Relinqs rtr
	RelinquishRetries uint64
	// Number of attribute changed requests seen
	AttributeChangedRequestsSeen uint64
	// Number of attr changed requests queued
	AttributeChangedRequestsQueued uint64
	// Number of attr changed rejected -ENOBUFS
	AttributeChangedRejectDueToEnobufs uint64
	// Number of attr changed failed -ENOMEM
	AttributeChangedFailedDueToEnomem uint64
	// Number of attr changed ops given CPU time
	AttributeChangedOps uint64
	// Number of allocation requests seen
	AllocationRequestsSeen uint64
	// Number of successful alloc reqs
	AllocationOkRequests uint64
	// Number of alloc reqs that waited on lookup completion
	AllocationWaitingOnLookup uint64
	// Number of alloc reqs rejected -ENOBUFS
	AllocationsRejectedDueToEnobufs uint64
	// Number of alloc reqs aborted -ERESTARTSYS
	AllocationsAbortedDueToErestartsys uint64
	// Number of alloc reqs submitted
	AllocationOperationsSubmitted uint64
	// Number of alloc reqs waited for CPU time
	AllocationsWaitedForCPU uint64
	// Number of alloc reqs aborted due to object death
	AllocationsAbortedDueToObjectDeath uint64
	// Number of retrieval (read) requests seen
	RetrievalsReadRequests uint64
	// Number of successful retr reqs
	RetrievalsOk uint64
	// Number of retr reqs that waited on lookup completion
	RetrievalsWaitingLookupCompletion uint64
	// Number of retr reqs returned -ENODATA
	RetrievalsReturnedEnodata uint64
	// Number of retr reqs rejected -ENOBUFS
	RetrievalsRejectedDueToEnobufs uint64
	// Number of retr reqs aborted -ERESTARTSYS
	RetrievalsAbortedDueToErestartsys uint64
	// Number of retr reqs failed -ENOMEM
	RetrievalsFailedDueToEnomem uint64
	// Number of retr reqs submitted
	RetrievalsRequests uint64
	// Number of retr reqs waited for CPU time
	RetrievalsWaitingCPU uint64
	// Number of retr reqs aborted due to object death
	RetrievalsAbortedDueToObjectDeath uint64
	// Number of storage (write) requests seen
	StoreWriteRequests uint64
	// Number of successful store reqs
	StoreSuccessfulRequests uint64
	// Number of store reqs on a page already pending storage
	StoreRequestsOnPendingStorage uint64
	// Number of store reqs rejected -ENOBUFS
	StoreRequestsRejectedDueToEnobufs uint64
	// Number of store reqs failed -ENOMEM
	StoreRequestsFailedDueToEnomem uint64
	// Number of store reqs submitted
	StoreRequestsSubmitted uint64
	// Number of store reqs granted CPU time
	StoreRequestsRunning uint64
	// Number of pages given store req processing time
	StorePagesWithRequestsProcessing uint64
	// Number of store reqs deleted from tracking tree
	StoreRequestsDeleted uint64
	// Number of store reqs over store limit
	StoreRequestsOverStoreLimit uint64
	// Number of release reqs against pages with no pending store
	ReleaseRequestsAgainstPagesWithNoPendingStorage uint64
	// Number of release reqs against pages stored by time lock granted
	ReleaseRequestsAgainstPagesStoredByTimeLockGranted uint64
	// Number of release reqs ignored due to in-progress store
	ReleaseRequestsIgnoredDueToInProgressStore uint64
	// Number of page stores cancelled due to release req
	PageStoresCancelledByReleaseRequests uint64
	VmscanWaiting                        uint64
	// Number of times async ops added to pending queues
	OpsPending uint64
	// Number of times async ops given CPU time
	OpsRunning uint64
	// Number of times async ops queued for processing
	OpsEnqueued uint64
	// Number of async ops cancelled
	OpsCancelled uint64
	// Number of async ops rejected due to object lookup/create failure
	OpsRejected uint64
	// Number of async ops initialised
	OpsInitialised uint64
	// Number of async ops queued for deferred release
	OpsDeferred uint64
	// Number of async ops released (should equal ini=N when idle)
	OpsReleased uint64
	// Number of deferred-release async ops garbage collected
	OpsGarbageCollected uint64
	// Number of in-progress alloc_object() cache ops
	CacheopAllocationsinProgress uint64
	// Number of in-progress lookup_object() cache ops
	CacheopLookupObjectInProgress uint64
	// Number of in-progress lookup_complete() cache ops
	CacheopLookupCompleteInPorgress uint64
	// Number of in-progress grab_object() cache ops
	CacheopGrabObjectInProgress uint64
	CacheopInvalidations        uint64
	// Number of in-progress update_object() cache ops
	CacheopUpdateObjectInProgress uint64
	// Number of in-progress drop_object() cache ops
	CacheopDropObjectInProgress uint64
	// Number of in-progress put_object() cache ops
	CacheopPutObjectInProgress uint64
	// Number of in-progress attr_changed() cache ops
	CacheopAttributeChangeInProgress uint64
	// Number of in-progress sync_cache() cache ops
	CacheopSyncCacheInProgress uint64
	// Number of in-progress read_or_alloc_page() cache ops
	CacheopReadOrAllocPageInProgress uint64
	// Number of in-progress read_or_alloc_pages() cache ops
	CacheopReadOrAllocPagesInProgress uint64
	// Number of in-progress allocate_page() cache ops
	CacheopAllocatePageInProgress uint64
	// Number of in-progress allocate_pages() cache ops
	CacheopAllocatePagesInProgress uint64
	// Number of in-progress write_page() cache ops
	CacheopWritePagesInProgress uint64
	// Number of in-progress uncache_page() cache ops
	CacheopUncachePagesInProgress uint64
	// Number of in-progress dissociate_pages() cache ops
	CacheopDissociatePagesInProgress uint64
	// Number of object lookups/creations rejected due to lack of space
	CacheevLookupsAndCreationsRejectedLackSpace uint64
	// Number of stale objects deleted
	CacheevStaleObjectsDeleted uint64
	// Number of objects retired when relinquished
	CacheevRetiredWhenReliquished uint64
	// Number of objects culled
	CacheevObjectsCulled uint64
}

// Fscacheinfo returns information about current fscache statistics.
// See https://www.kernel.org/doc/Documentation/filesystems/caching/fscache.txt
func (fs FS) Fscacheinfo() (Fscacheinfo, error) {
	b, err := util.ReadFileNoStat(fs.proc.Path("fs/fscache/stats"))
	if err != nil {
		return Fscacheinfo{}, err
	}

	m, err := parseFscacheinfo(bytes.NewReader(b))
	if err != nil {
		return Fscacheinfo{}, fmt.Errorf("failed to parse Fscacheinfo: %w", err)
	}

	return *m, nil
}

func setFSCacheFields(fields []string, setFields ...*uint64) error {
	var err error
	if len(fields) < len(setFields) {
		return fmt.Errorf("Insufficient number of fields, expected %v, got %v", len(setFields), len(fields))
	}

	for i := range setFields {
		*setFields[i], err = strconv.ParseUint(strings.Split(fields[i], "=")[1], 0, 64)
		if err != nil {
			return err
		}
	}
	return nil
}

func parseFscacheinfo(r io.Reader) (*Fscacheinfo, error) {
	var m Fscacheinfo
	s := bufio.NewScanner(r)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) < 2 {
			return nil, fmt.Errorf("malformed Fscacheinfo line: %q", s.Text())
		}

		switch fields[0] {
		case "Cookies:":
			err := setFSCacheFields(fields[1:], &m.IndexCookiesAllocated, &m.DataStorageCookiesAllocated,
				&m.SpecialCookiesAllocated)
			if err != nil {
				return &m, err
			}
		case "Objects:":
			err := setFSCacheFields(fields[1:], &m.ObjectsAllocated, &m.ObjectAllocationsFailure,
				&m.ObjectsAvailable, &m.ObjectsDead)
			if err != nil {
				return &m, err
			}
		case "ChkAux":
			err := setFSCacheFields(fields[2:], &m.ObjectsWithoutCoherencyCheck, &m.ObjectsWithCoherencyCheck,
				&m.ObjectsNeedCoherencyCheckUpdate, &m.ObjectsDeclaredObsolete)
			if err != nil {
				return &m, err
			}
		case "Pages":
			err := setFSCacheFields(fields[2:], &m.PagesMarkedAsBeingCached, &m.UncachePagesRequestSeen)
			if err != nil {
				return &m, err
			}
		case "Acquire:":
			err := setFSCacheFields(fields[1:], &m.AcquireCookiesRequestSeen, &m.AcquireRequestsWithNullParent,
				&m.AcquireRequestsRejectedNoCacheAvailable, &m.AcquireRequestsSucceeded, &m.AcquireRequestsRejectedDueToError,
				&m.AcquireRequestsFailedDueToEnomem)
			if err != nil {
				return &m, err
			}
		case "Lookups:":
			err := setFSCacheFields(fields[1:], &m.LookupsNumber, &m.LookupsNegative, &m.LookupsPositive,
				&m.ObjectsCreatedByLookup, &m.LookupsTimedOutAndRequed)
			if err != nil {
				return &m, err
			}
		case "Invals":
			err := setFSCacheFields(fields[2:], &m.InvalidationsNumber, &m.InvalidationsRunning)
			if err != nil {
				return &m, err
			}
		case "Updates:":
			err := setFSCacheFields(fields[1:], &m.UpdateCookieRequestSeen, &m.UpdateRequestsWithNullParent,
				&m.UpdateRequestsRunning)
			if err != nil {
				return &m, err
			}
		case "Relinqs:":
			err := setFSCacheFields(fields[1:], &m.RelinquishCookiesRequestSeen, &m.RelinquishCookiesWithNullParent,
				&m.RelinquishRequestsWaitingCompleteCreation, &m.RelinquishRetries)
			if err != nil {
				return &m, err
			}
		case "AttrChg:":
			err := setFSCacheFields(fields[1:], &m.AttributeChangedRequestsSeen, &m.AttributeChangedRequestsQueued,
				&m.AttributeChangedRejectDueToEnobufs, &m.AttributeChangedFailedDueToEnomem, &m.AttributeChangedOps)
			if err != nil {
				return &m, err
			}
		case "Allocs":
			if strings.Split(fields[2], "=")[0] == "n" {
				err := setFSCacheFields(fields[2:], &m.AllocationRequestsSeen, &m.AllocationOkRequests,
					&m.AllocationWaitingOnLookup, &m.AllocationsRejectedDueToEnobufs, &m.AllocationsAbortedDueToErestartsys)
				if err != nil {
					return &m, err
				}
			} else {
				err := setFSCacheFields(fields[2:], &m.AllocationOperationsSubmitted, &m.AllocationsWaitedForCPU,
					&m.AllocationsAbortedDueToObjectDeath)
				if err != nil {
					return &m, err
				}
			}
		case "Retrvls:":
			if strings.Split(fields[1], "=")[0] == "n" {
				err := setFSCacheFields(fields[1:], &m.RetrievalsReadRequests, &m.RetrievalsOk, &m.RetrievalsWaitingLookupCompletion,
					&m.RetrievalsReturnedEnodata, &m.RetrievalsRejectedDueToEnobufs, &m.RetrievalsAbortedDueToErestartsys,
					&m.RetrievalsFailedDueToEnomem)
				if err != nil {
					return &m, err
				}
			} else {
				err := setFSCacheFields(fields[1:], &m.RetrievalsRequests, &m.RetrievalsWaitingCPU, &m.RetrievalsAbortedDueToObjectDeath)
				if err != nil {
					return &m, err
				}
			}
		case "Stores":
			if strings.Split(fields[2], "=")[0] == "n" {
				err := setFSCacheFields(fields[2:], &m.StoreWriteRequests, &m.StoreSuccessfulRequests,
					&m.StoreRequestsOnPendingStorage, &m.StoreRequestsRejectedDueToEnobufs, &m.StoreRequestsFailedDueToEnomem)
				if err != nil {
					return &m, err
				}
			} else {
				err := setFSCacheFields(fields[2:], &m.StoreRequestsSubmitted, &m.StoreRequestsRunning,
					&m.StorePagesWithRequestsProcessing, &m.StoreRequestsDeleted, &m.StoreRequestsOverStoreLimit)
				if err != nil {
					return &m, err
				}
			}
		case "VmScan":
			err := setFSCacheFields(fields[2:], &m.ReleaseRequestsAgainstPagesWithNoPendingStorage,
				&m.ReleaseRequestsAgainstPagesStoredByTimeLockGranted, &m.ReleaseRequestsIgnoredDueToInProgressStore,
				&m.PageStoresCancelledByReleaseRequests, &m.VmscanWaiting)
			if err != nil {
				return &m, err
			}
		case "Ops":
			if strings.Split(fields[2], "=")[0] == "pend" {
				err := setFSCacheFields(fields[2:], &m.OpsPending, &m.OpsRunning, &m.OpsEnqueued, &m.OpsCancelled, &m.OpsRejected)
				if err != nil {
					return &m, err
				}
			} else {
				err := setFSCacheFields(fields[2:], &m.OpsInitialised, &m.OpsDeferred, &m.OpsReleased, &m.OpsGarbageCollected)
				if err != nil {
					return &m, err
				}
			}
		case "CacheOp:":
			if strings.Split(fields[1], "=")[0] == "alo" {
				err := setFSCacheFields(fields[1:], &m.CacheopAllocationsinProgress, &m.CacheopLookupObjectInProgress,
					&m.CacheopLookupCompleteInPorgress, &m.CacheopGrabObjectInProgress)
				if err != nil {
					return &m, err
				}
			} else if strings.Split(fields[1], "=")[0] == "inv" {
				err := setFSCacheFields(fields[1:], &m.CacheopInvalidations, &m.CacheopUpdateObjectInProgress,
					&m.CacheopDropObjectInProgress, &m.CacheopPutObjectInProgress, &m.CacheopAttributeChangeInProgress,
					&m.CacheopSyncCacheInProgress)
				if err != nil {
					return &m, err
				}
			} else {
				err := setFSCacheFields(fields[1:], &m.CacheopReadOrAllocPageInProgress, &m.CacheopReadOrAllocPagesInProgress,
					&m.CacheopAllocatePageInProgress, &m.CacheopAllocatePagesInProgress, &m.CacheopWritePagesInProgress,
					&m.CacheopUncachePagesInProgress, &m.CacheopDissociatePagesInProgress)
				if err != nil {
					return &m, err
				}
			}
		case "CacheEv:":
			err := setFSCacheFields(fields[1:], &m.CacheevLookupsAndCreationsRejectedLackSpace, &m.CacheevStaleObjectsDeleted,
				&m.CacheevRetiredWhenReliquished, &m.CacheevObjectsCulled)
			if err != nil {
				return &m, err
			}
		}
	}

	return &m, nil
}
