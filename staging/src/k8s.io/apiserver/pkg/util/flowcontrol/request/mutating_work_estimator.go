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
	"time"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func newMutatingWorkEstimator(countFn watchCountGetterFunc) WorkEstimatorFunc {
	estimator := &mutatingWorkEstimator{
		countFn: countFn,
	}
	return estimator.estimate
}

type mutatingWorkEstimator struct {
	countFn watchCountGetterFunc
}

const (
	watchesPerSeat          = 10.0
	eventAdditionalDuration = 5 * time.Millisecond
)

func (e *mutatingWorkEstimator) estimate(r *http.Request) WorkEstimate {
	// TODO(wojtekt): Remove once we tune the algorithm to not fail
	// scalability tests.
	return WorkEstimate{
		InitialSeats: 1,
	}

	requestInfo, ok := apirequest.RequestInfoFrom(r.Context())
	if !ok {
		// no RequestInfo should never happen, but to be on the safe side
		// let's return a large value.
		return WorkEstimate{
			InitialSeats:      1,
			FinalSeats:        maximumSeats,
			AdditionalLatency: eventAdditionalDuration,
		}
	}
	watchCount := e.countFn(requestInfo)

	// The cost of the request associated with the watchers of that event
	// consists of three parts:
	// - cost of going through the event change logic
	// - cost of serialization of the event
	// - cost of processing an event object for each watcher (e.g. filtering,
	//     sending data over network)
	// We're starting simple to get some operational experience with it and
	// we will work on tuning the algorithm later. As a starting point we
	// we simply assume that processing 1 event takes 1/Nth of a seat for
	// M milliseconds and processing different events is infinitely parallelizable.
	// We simply record the appropriate values here and rely on potential
	// reshaping of the request if the concurrency limit for a given priority
	// level will not allow to run request with that many seats.
	//
	// TODO: As described in the KEP, we should take into account that not all
	//   events are equal and try to estimate the cost of a single event based on
	//   some historical data about size of events.
	var finalSeats uint
	var additionalLatency time.Duration

	// TODO: Make this unconditional after we tune the algorithm better.
	//   Technically, there is an overhead connected to processing an event after
	//   the request finishes even if there is a small number of watches.
	//   However, until we tune the estimation we want to stay on the safe side
	//   an avoid introducing additional latency for almost every single request.
	if watchCount >= watchesPerSeat {
		finalSeats = uint(math.Ceil(float64(watchCount) / watchesPerSeat))
		additionalLatency = eventAdditionalDuration
	}

	return WorkEstimate{
		InitialSeats:      1,
		FinalSeats:        finalSeats,
		AdditionalLatency: additionalLatency,
	}
}
