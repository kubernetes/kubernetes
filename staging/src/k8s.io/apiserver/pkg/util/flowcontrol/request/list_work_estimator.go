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
	"net/url"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
)

func newListWorkEstimator(countFn objectCountGetterFunc, config *WorkEstimatorConfig) WorkEstimatorFunc {
	estimator := &listWorkEstimator{
		config:        config,
		countGetterFn: countFn,
	}
	return estimator.estimate
}

type listWorkEstimator struct {
	config        *WorkEstimatorConfig
	countGetterFn objectCountGetterFunc
}

func (e *listWorkEstimator) estimate(r *http.Request, flowSchemaName, priorityLevelName string) WorkEstimate {
	requestInfo, ok := apirequest.RequestInfoFrom(r.Context())
	if !ok {
		// no RequestInfo should never happen, but to be on the safe side
		// let's return maximumSeats
		return WorkEstimate{InitialSeats: e.config.MaximumSeats}
	}

	query := r.URL.Query()
	listOptions := metav1.ListOptions{}
	if err := metav1.Convert_url_Values_To_v1_ListOptions(&query, &listOptions, nil); err != nil {
		klog.ErrorS(err, "Failed to convert options while estimating work for the list request")

		// This request is destined to fail in the validation layer,
		// return maximumSeats for this request to be consistent.
		return WorkEstimate{InitialSeats: e.config.MaximumSeats}
	}
	isListFromCache := !shouldListFromStorage(query, &listOptions)

	numStored, err := e.countGetterFn(key(requestInfo))
	switch {
	case err == ObjectCountStaleErr:
		// object count going stale is indicative of degradation, so we should
		// be conservative here and allocate maximum seats to this list request.
		// NOTE: if a CRD is removed, its count will go stale first and then the
		// pruner will eventually remove the CRD from the cache.
		return WorkEstimate{InitialSeats: e.config.MaximumSeats}
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
		return WorkEstimate{InitialSeats: e.config.MinimumSeats}
	case err != nil:
		// we should never be here since Get returns either ObjectCountStaleErr or
		// ObjectCountNotFoundErr, return maximumSeats to be on the safe side.
		klog.ErrorS(err, "Unexpected error from object count tracker")
		return WorkEstimate{InitialSeats: e.config.MaximumSeats}
	}

	limit := numStored
	if utilfeature.DefaultFeatureGate.Enabled(features.APIListChunking) && listOptions.Limit > 0 &&
		listOptions.Limit < numStored {
		limit = listOptions.Limit
	}

	var estimatedObjectsToBeProcessed int64

	switch {
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
	seats := uint64(math.Ceil(float64(estimatedObjectsToBeProcessed) / e.config.ObjectsPerSeat))

	// make sure we never return a seat of zero
	if seats < e.config.MinimumSeats {
		seats = e.config.MinimumSeats
	}
	if seats > e.config.MaximumSeats {
		seats = e.config.MaximumSeats
	}
	return WorkEstimate{InitialSeats: seats}
}

func key(requestInfo *apirequest.RequestInfo) string {
	groupResource := &schema.GroupResource{
		Group:    requestInfo.APIGroup,
		Resource: requestInfo.Resource,
	}
	return groupResource.String()
}

// NOTICE: Keep in sync with shouldDelegateList function in
//
//	staging/src/k8s.io/apiserver/pkg/storage/cacher/cacher.go
func shouldListFromStorage(query url.Values, opts *metav1.ListOptions) bool {
	resourceVersion := opts.ResourceVersion
	pagingEnabled := utilfeature.DefaultFeatureGate.Enabled(features.APIListChunking)
	hasContinuation := pagingEnabled && len(opts.Continue) > 0
	hasLimit := pagingEnabled && opts.Limit > 0 && resourceVersion != "0"
	return resourceVersion == "" || hasContinuation || hasLimit || opts.ResourceVersionMatch == metav1.ResourceVersionMatchExact
}
