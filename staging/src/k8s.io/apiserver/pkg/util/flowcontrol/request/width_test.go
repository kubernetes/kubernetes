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
	"errors"
	"net/http"
	"testing"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestWorkEstimator(t *testing.T) {
	tests := []struct {
		name                 string
		requestURI           string
		requestInfo          *apirequest.RequestInfo
		counts               map[string]int64
		countErr             error
		initialSeatsExpected uint
	}{
		{
			name:                 "request has no RequestInfo",
			requestURI:           "http://server/apis/",
			requestInfo:          nil,
			initialSeatsExpected: maximumSeats,
		},
		{
			name:       "request verb is not list",
			requestURI: "http://server/apis/",
			requestInfo: &apirequest.RequestInfo{
				Verb: "get",
			},
			initialSeatsExpected: minimumSeats,
		},
		{
			name:       "request verb is list, conversion to ListOptions returns error",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=invalid",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 799,
			},
			initialSeatsExpected: maximumSeats,
		},
		{
			name:       "request verb is list, has limit and resource version is 1",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=399&resourceVersion=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 699,
			},
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, limit not set",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 699,
			},
			initialSeatsExpected: 7,
		},
		{
			name:       "request verb is list, resource version not set",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=399",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 699,
			},
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, no query parameters, count known",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 399,
			},
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, no query parameters, count not known",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			countErr:             ObjectCountNotFoundErr,
			initialSeatsExpected: maximumSeats,
		},
		{
			name:       "request verb is list, continuation is set",
			requestURI: "http://server/apis/foo.bar/v1/events?continue=token&limit=399",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 699,
			},
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, resource version is zero",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=299&resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 399,
			},
			initialSeatsExpected: 4,
		},
		{
			name:       "request verb is list, resource version is zero, no limit",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 799,
			},
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, resource version match is Exact",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=foo&resourceVersionMatch=Exact&limit=399",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 699,
			},
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, resource version match is NotOlderThan, limit not specified",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=foo&resourceVersionMatch=NotOlderThan",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 799,
			},
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, maximum is capped",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=foo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 1999,
			},
			initialSeatsExpected: maximumSeats,
		},
		{
			name:       "request verb is list, list from cache, count not known",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0&limit=799",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			countErr:             ObjectCountNotFoundErr,
			initialSeatsExpected: maximumSeats,
		},
		{
			name:       "request verb is list, object count is stale",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			counts: map[string]int64{
				"events.foo.bar": 799,
			},
			countErr:             ObjectCountStaleErr,
			initialSeatsExpected: maximumSeats,
		},
		{
			name:       "request verb is list, object count is not found",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			countErr:             ObjectCountNotFoundErr,
			initialSeatsExpected: maximumSeats,
		},
		{
			name:       "request verb is list, count getter throws unknown error",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			countErr:             errors.New("unknown error"),
			initialSeatsExpected: maximumSeats,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			counts := test.counts
			if len(counts) == 0 {
				counts = map[string]int64{}
			}
			countsFn := func(key string) (int64, error) {
				return counts[key], test.countErr
			}
			estimator := NewWorkEstimator(countsFn)

			req, err := http.NewRequest("GET", test.requestURI, nil)
			if err != nil {
				t.Fatalf("Failed to create new HTTP request - %v", err)
			}

			if test.requestInfo != nil {
				req = req.WithContext(apirequest.WithRequestInfo(req.Context(), test.requestInfo))
			}

			workestimateGot := estimator.EstimateWork(req)
			if test.initialSeatsExpected != workestimateGot.InitialSeats {
				t.Errorf("Expected work estimate to match: %d seats, but got: %d seats", test.initialSeatsExpected, workestimateGot.InitialSeats)
			}
		})
	}
}
