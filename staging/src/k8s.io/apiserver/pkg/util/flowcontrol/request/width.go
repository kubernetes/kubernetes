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
	"net/http"
)

type WorkEstimate struct {
	// Seats represents the number of seats associated with this request
	Seats uint
}

// DefaultWorkEstimator returns estimation with default number of seats
// of 1.
//
// TODO: when we plumb in actual work estimate handling for different
//  type of request(s) this function will iterate through a chain
//  of workEstimator instance(s).
func DefaultWorkEstimator(_ *http.Request) WorkEstimate {
	return WorkEstimate{
		Seats: 1,
	}
}

// WorkEstimatorFunc returns the estimated work of a given request.
// This function will be used by the Priority & Fairness filter to
// estimate the work of of incoming requests.
type WorkEstimatorFunc func(*http.Request) WorkEstimate

func (e WorkEstimatorFunc) EstimateWork(r *http.Request) WorkEstimate {
	return e(r)
}
