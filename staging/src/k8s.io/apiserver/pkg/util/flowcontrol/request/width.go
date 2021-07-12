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

type Width struct {
	// Seats represents the number of seats associated with this request
	Seats uint
}

// DefaultWidthEstimator returns returns '1' as the "width"
// of the given request.
//
// TODO: when we plumb in actual "width" handling for different
//  type of request(s) this function will iterate through a chain
//  of widthEstimator instance(s).
func DefaultWidthEstimator(_ *http.Request) Width {
	return Width{
		Seats: 1,
	}
}

// WidthEstimatorFunc returns the estimated "width" of a given request.
// This function will be used by the Priority & Fairness filter to
// estimate the "width" of incoming requests.
type WidthEstimatorFunc func(*http.Request) Width

func (e WidthEstimatorFunc) EstimateWidth(r *http.Request) Width {
	return e(r)
}
