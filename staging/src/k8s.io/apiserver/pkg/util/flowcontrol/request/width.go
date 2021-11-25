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
	"fmt"
	"net/http"
	"time"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

const (
	// the minimum number of seats a request must occupy
	minimumSeats = 1

	// the maximum number of seats a request can occupy
	maximumSeats = 10
)

// WorkEstimate carries three of the four parameters that determine the work in a request.
// The fourth parameter is the duration of the initial phase of execution.
type WorkEstimate struct {
	// InitialSeats is the number of seats occupied while the server is
	// executing this request.
	InitialSeats uint

	// FinalSeats is the number of seats occupied at the end,
	// during the AdditionalLatency.
	FinalSeats uint

	// AdditionalLatency specifies the additional duration the seats allocated
	// to this request must be reserved after the given request had finished.
	// AdditionalLatency should not have any impact on the user experience, the
	// caller must not experience this additional latency.
	AdditionalLatency time.Duration
}

// MaxSeats returns the maximum number of seats the request occupies over the
// phases of being served.
func (we *WorkEstimate) MaxSeats() int {
	if we.InitialSeats >= we.FinalSeats {
		return int(we.InitialSeats)
	}

	return int(we.FinalSeats)
}

// objectCountGetterFunc represents a function that gets the total
// number of objects for a given resource.
type objectCountGetterFunc func(string) (int64, error)

// watchCountGetterFunc represents a function that gets the total
// number of watchers potentially interested in a given request.
type watchCountGetterFunc func(*apirequest.RequestInfo) int

// NewWorkEstimator estimates the work that will be done by a given request,
// if no WorkEstimatorFunc matches the given request then the default
// work estimate of 1 seat is allocated to the request.
func NewWorkEstimator(objectCountFn objectCountGetterFunc, watchCountFn watchCountGetterFunc) WorkEstimatorFunc {
	estimator := &workEstimator{
		listWorkEstimator:     newListWorkEstimator(objectCountFn),
		mutatingWorkEstimator: newMutatingWorkEstimator(watchCountFn),
	}
	return estimator.estimate
}

// WorkEstimatorFunc returns the estimated work of a given request.
// This function will be used by the Priority & Fairness filter to
// estimate the work of of incoming requests.
type WorkEstimatorFunc func(request *http.Request, flowSchemaName, priorityLevelName string) WorkEstimate

func (e WorkEstimatorFunc) EstimateWork(r *http.Request, flowSchemaName, priorityLevelName string) WorkEstimate {
	return e(r, flowSchemaName, priorityLevelName)
}

type workEstimator struct {
	// listWorkEstimator estimates work for list request(s)
	listWorkEstimator WorkEstimatorFunc
	// mutatingWorkEstimator calculates the width of mutating request(s)
	mutatingWorkEstimator WorkEstimatorFunc
}

func (e *workEstimator) estimate(r *http.Request, flowSchemaName, priorityLevelName string) WorkEstimate {
	requestInfo, ok := apirequest.RequestInfoFrom(r.Context())
	if !ok {
		klog.ErrorS(fmt.Errorf("no RequestInfo found in context"), "Failed to estimate work for the request", "URI", r.RequestURI)
		// no RequestInfo should never happen, but to be on the safe side let's return maximumSeats
		return WorkEstimate{InitialSeats: maximumSeats}
	}

	switch requestInfo.Verb {
	case "list":
		return e.listWorkEstimator.EstimateWork(r, flowSchemaName, priorityLevelName)
	case "create", "update", "patch", "delete":
		return e.mutatingWorkEstimator.EstimateWork(r, flowSchemaName, priorityLevelName)
	}

	return WorkEstimate{InitialSeats: minimumSeats}
}
