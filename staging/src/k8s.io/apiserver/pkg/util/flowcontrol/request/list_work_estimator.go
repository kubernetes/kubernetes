/*
Copyright 2021 The Kubernetes Authors.

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

package request

import (
	"math"
	"net/http"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/delegator"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
)

const (
	bytesPerSeat                     = 100_000
	cacheWithStreamingMaxMemoryUsage = 1_000_000
	// 1.5MB is the recommended client request size in byte
	// the etcd server should accept. See
	// https://github.com/etcd-io/etcd/blob/release-3.4/embed/config.go#L56.
	maxObjectSize       = 1_500_000
	infiniteObjectCount = 1_000_000_000
)

func newListWorkEstimator(countFn statsGetterFunc, config *WorkEstimatorConfig, maxSeatsFn maxSeatsFunc) *listWorkEstimator {
	estimator := &listWorkEstimator{
		config:        config,
		statsGetterFn: countFn,
		maxSeatsFn:    maxSeatsFn,
	}
	return estimator
}

type listWorkEstimator struct {
	config        *WorkEstimatorConfig
	statsGetterFn statsGetterFunc
	maxSeatsFn    maxSeatsFunc
}

func (e *listWorkEstimator) estimate(r *http.Request, flowSchemaName, priorityLevelName string) WorkEstimate {
	minSeats := e.config.MinimumSeats
	maxSeats := e.maxSeatsFn(priorityLevelName)
	if maxSeats == 0 || maxSeats > e.config.MaximumListSeatsLimit {
		maxSeats = e.config.MaximumListSeatsLimit
	}

	requestInfo, ok := apirequest.RequestInfoFrom(r.Context())
	if !ok {
		// no RequestInfo should never happen, but to be on the safe side
		// let's return maximumSeats
		return WorkEstimate{InitialSeats: maxSeats}
	}

	// Requests with metadata.name specified are usually executed as get
	// requests in storage layer so their width should be 1.
	// Example of such list requests:
	// /apis/certificates.k8s.io/v1/certificatesigningrequests?fieldSelector=metadata.name%3Dcsr-xxs4m
	// /api/v1/namespaces/test/configmaps?fieldSelector=metadata.name%3Dbig-deployment-1&limit=500&resourceVersion=0
	matchesSingle := requestInfo.Name != ""

	query := r.URL.Query()
	listOptions := metav1.ListOptions{}
	if err := metav1.Convert_url_Values_To_v1_ListOptions(&query, &listOptions, nil); err != nil {
		klog.ErrorS(err, "Failed to convert options while estimating work for the list request")

		// This request is destined to fail in the validation layer,
		// return maximumSeats for this request to be consistent.
		return WorkEstimate{InitialSeats: maxSeats}
	}

	// For watch requests we want to set the cost low if they aren't requesting for init events,
	// either via the explicit SendInitialEvents param or via legacy watches that have RV=0 or unset.
	if requestInfo.Verb == "watch" {
		sendInitEvents := listOptions.SendInitialEvents != nil && *listOptions.SendInitialEvents
		legacyWatch := listOptions.ResourceVersion == "" || listOptions.ResourceVersion == "0"
		if !sendInitEvents && !legacyWatch {
			return WorkEstimate{InitialSeats: e.config.MinimumSeats}
		}
	}
	// TODO: Check whether watchcache is enabled.
	var listFromStorage bool
	result, err := delegator.ShouldDelegateListMeta(&listOptions, delegator.CacheWithoutSnapshots{})
	if err != nil {
		// Assume worse case where we need to reach to etcd.
		listFromStorage = true
	} else {
		listFromStorage = result.ShouldDelegate
	}
	isListFromCache := requestInfo.Verb == "watch" || !listFromStorage

	stats, err := e.statsGetterFn(key(requestInfo))
	switch {
	case err == ObjectCountStaleErr:
		// object count going stale is indicative of degradation, so we should
		// be conservative here and return maximum object count and size.
		// NOTE: if a CRD is removed, its count will go stale first and then the
		// pruner will eventually remove the CRD from the cache.
		stats = storage.Stats{ObjectCount: infiniteObjectCount, EstimatedAverageObjectSizeBytes: maxObjectSize}
	case err == ObjectCountNotFoundErr:
		// there are multiple scenarios in which we can see this error:
		//  a. the type is truly unknown, a typo on the caller's part.
		//  b. the count has gone stale for too long and the pruner
		//     has removed the type from the cache.
		//  c. the type is an aggregated resource that is served by a
		//     different apiserver (thus its object count is not updated)
		// we don't have a way to distinguish between those situations.
		// However, in case c, the request is delegated to a different apiserver,
		// and thus its cost for our server is minimal. To avoid the situation
		// when aggregated API calls are overestimated, we allocate the minimum
		// possible seats (see #109106 as an example when being more conservative
		// led to problems).
		return WorkEstimate{InitialSeats: minSeats}
	case err != nil:
		// we should never be here since Get returns either ObjectCountStaleErr or
		// ObjectCountNotFoundErr, return maximum object count and size.
		klog.ErrorS(err, "Unexpected error from object count tracker")
		stats = storage.Stats{ObjectCount: infiniteObjectCount, EstimatedAverageObjectSizeBytes: maxObjectSize}
	}

	var seats uint64
	if utilfeature.DefaultFeatureGate.Enabled(features.SizeBasedListCostEstimate) {
		seats = e.seatsBasedOnObjectSize(stats, listOptions, isListFromCache, matchesSingle)
	} else {
		seats = e.seatsBasedOnObjectCount(stats, listOptions, isListFromCache, matchesSingle)
	}

	// make sure we never return a seat of zero
	if seats < minSeats {
		seats = minSeats
	}
	if seats > maxSeats {
		seats = maxSeats
	}
	return WorkEstimate{InitialSeats: seats}
}

func (e *listWorkEstimator) seatsBasedOnObjectCount(stats storage.Stats, listOptions metav1.ListOptions, isListFromCache bool, matchesSingle bool) uint64 {
	numStored := stats.ObjectCount
	limit := numStored
	if listOptions.Limit > 0 && listOptions.Limit < numStored {
		limit = listOptions.Limit
	}

	var estimatedObjectsToBeProcessed int64

	switch {
	case matchesSingle:
		estimatedObjectsToBeProcessed = 1
	case isListFromCache:
		// TODO: For resources that implement indexes at the watchcache level,
		//  we need to adjust the cost accordingly
		estimatedObjectsToBeProcessed = numStored
	case listOptions.FieldSelector != "" || listOptions.LabelSelector != "":
		estimatedObjectsToBeProcessed = numStored + limit
	default:
		estimatedObjectsToBeProcessed = 2 * limit
	}

	// for now, our rough estimate is to allocate one seat to each 100 obejcts that
	// will be processed by the list request.
	// we will come up with a different formula for the transformation function and/or
	// fine tune this number in future iteratons.
	return uint64(math.Ceil(float64(estimatedObjectsToBeProcessed) / e.config.ObjectsPerSeat))
}

func (e *listWorkEstimator) seatsBasedOnObjectSize(stats storage.Stats, listOptions metav1.ListOptions, isListFromCache bool, matchesSingle bool) uint64 {
	if stats.EstimatedAverageObjectSizeBytes <= 0 && stats.ObjectCount != 0 {
		stats.EstimatedAverageObjectSizeBytes = maxObjectSize
	}
	limited := stats.ObjectCount
	if listOptions.Limit > 0 && listOptions.Limit < limited {
		limited = listOptions.Limit
	}
	var objectsLoadedInMemory int64
	switch {
	case matchesSingle:
		objectsLoadedInMemory = 1
	case isListFromCache:
		objectsLoadedInMemory = limited
	case listOptions.FieldSelector != "" || listOptions.LabelSelector != "":
		objectsLoadedInMemory = max(limited, stats.ObjectCount/2)
	default:
		objectsLoadedInMemory = limited
	}

	memoryUsedAtOnce := objectsLoadedInMemory * stats.EstimatedAverageObjectSizeBytes
	if isListFromCache {
		// TODO: Identify if the resource is streamed
		memoryUsedAtOnce = min(memoryUsedAtOnce, cacheWithStreamingMaxMemoryUsage)
	}
	return uint64(math.Ceil(float64(memoryUsedAtOnce) / bytesPerSeat))
}

func key(requestInfo *apirequest.RequestInfo) string {
	groupResource := &schema.GroupResource{
		Group:    requestInfo.APIGroup,
		Resource: requestInfo.Resource,
	}
	return groupResource.String()
}
