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
	"time"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestWorkEstimator(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)

	defaultCfg := DefaultWorkEstimatorConfig()

	tests := []struct {
		name                      string
		requestURI                string
		requestInfo               *apirequest.RequestInfo
		stats                     storage.Stats
		statsErr                  error
		watchCount                int
		maxSeats                  uint64
		initialSeatsExpected      uint64
		finalSeatsExpected        uint64
		additionalLatencyExpected time.Duration
	}{
		{
			name:                 "request has no RequestInfo, expect maxSeats",
			requestURI:           "http://server/apis/",
			requestInfo:          nil,
			maxSeats:             100,
			initialSeatsExpected: 100,
		},
		{
			name:       "request verb is not list, expect minSeats",
			requestURI: "http://server/apis/",
			requestInfo: &apirequest.RequestInfo{
				Verb: "get",
			},
			maxSeats:             10,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, conversion to ListOptions returns error, expect maxSeats",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=invalid",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 100,
		},
		{
			name:       "request verb is list, resource version 1, limit 399, expect read 4MB read from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=399&resourceVersion=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 699, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 40,
		},
		{
			name:       "request verb is list, resource version 1, expect seats capped by cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 699, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 10,
		},
		{
			name:       "request verb is list, resource version 1, expect read 7MB read from cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 69, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 7,
		},
		{
			name:       "request verb is list, limit 399, expect 4MB read from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=399",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 699, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 40,
		},
		{
			name:       "request verb is list, expect read 4MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 399, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 40,
		},
		{
			name:       "request verb is list, count not known, expect minSeats",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             ObjectCountNotFoundErr,
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, continuation is set, limit 399, expect read 4MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?continue=token&limit=399",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 699, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 40,
		},
		{
			name:       "request verb is list, resource version is zero, limit 299, expect seats capped by cache",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=299&resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 399, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 10,
		},
		{
			name:       "request verb is list, resource version is zero, limit 10, expect read 400KB from cache",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=20&resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 399, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 2,
		},
		{
			name:       "request verb is list, resource version is zero, expect seats capped by cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 10,
		},
		{
			name:       "request verb is list, resource version is zero, expect read 8MB from cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 79, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, resource version 1, match is Exact, expect read 4MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1&resourceVersionMatch=Exact&limit=399",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 699, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 40,
		},
		{
			name:       "request verb is list, resource version 1, match is NotOlderThan, expect seats capped by cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1&resourceVersionMatch=NotOlderThan",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 10,
		},
		{
			name:       "request verb is list, resource version 1, match is NotOlderThan, expect read 8 MB from cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1&resourceVersionMatch=NotOlderThan",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 79, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is list, resource version 1, match is Exact, expect seats capped by max",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1&resourceVersionMatch=Exact",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 5000, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             20,
			initialSeatsExpected: 20,
		},
		{
			name:       "request verb is list, bad match, expect read 2MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersionMatch=foo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 200, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 20,
		},
		{
			name:       "request verb is list, bad match, limit 399, expect read 4MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=foo&resourceVersionMatch=Exact&limit=399",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 699, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 40,
		},
		{
			name:       "request verb is list, resource version 1, match exact, expect seats capped by max seats",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=1&resourceVersionMatch=Exact",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 5000, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             20,
			initialSeatsExpected: 20,
		},
		{
			name:       "request verb is list, resource version 0, count is not found, expect min seats",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             ObjectCountNotFoundErr,
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, object count is stale, expect max seats",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 1, EstimatedAverageObjectSizeBytes: 1},
			statsErr:             ObjectCountStaleErr,
			maxSeats:             100,
			initialSeatsExpected: 100,
		},
		{
			name:       "request verb is list, no object size, expect seats capped by cache",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 1},
			maxSeats:             100,
			initialSeatsExpected: 15,
		},
		{
			name:       "request verb is list, object count is not found, expect min seats",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             ObjectCountNotFoundErr,
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, count getter throws unknown error, expect max seats",
			requestURI: "http://server/apis/foo.bar/v1/events",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             errors.New("unknown error"),
			maxSeats:             100,
			initialSeatsExpected: 100,
		},
		{
			name:       "request verb is list, resource version 0, count not known, limit 1, expect min seats",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0&limit=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             ObjectCountNotFoundErr,
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, object count is stale, limit 1, expect read max object size from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 1, EstimatedAverageObjectSizeBytes: 10_000},
			statsErr:             ObjectCountStaleErr,
			maxSeats:             100,
			initialSeatsExpected: 15,
		},
		{
			name:       "request verb is list, no object size, limit 1, expect read max object size from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events&limit=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 1},
			maxSeats:             100,
			initialSeatsExpected: 15,
		},
		{
			name:       "request verb is list, object count is not found, limit 1, expect min seats",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             ObjectCountNotFoundErr,
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, count getter throws unknown error, limit 1, expect read max object size from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=1",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             errors.New("unknown error"),
			maxSeats:             100,
			initialSeatsExpected: 15,
		},
		{
			name:       "request verb is list, resource version 0, count not known, limit 499, expect min seats",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0&limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             ObjectCountNotFoundErr,
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, object count is stale, limit 499, expect max seats",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 1, EstimatedAverageObjectSizeBytes: 1},
			statsErr:             ObjectCountStaleErr,
			maxSeats:             100,
			initialSeatsExpected: 100,
		},
		{
			name:       "request verb is list, no object size, limit 499, expect read max object size from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 1},
			maxSeats:             100,
			initialSeatsExpected: 15,
		},
		{
			name:       "request verb is list, no object size, resource version 0, limit 499, expect capped by cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0&limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 1},
			maxSeats:             100,
			initialSeatsExpected: 10,
		},
		{
			name:       "request verb is list, object count is not found, limit 499, expect min seats",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             ObjectCountNotFoundErr,
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is list, count getter throws unknown error, limit 499, expect max seats",
			requestURI: "http://server/apis/foo.bar/v1/events?limit=499",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			statsErr:             errors.New("unknown error"),
			maxSeats:             100,
			initialSeatsExpected: 100,
		},
		{
			name:       "request verb is list, metadata.name specified, expect read 200KB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?fieldSelector=metadata.name%3Dtest",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				Name:     "test",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 200_000},
			maxSeats:             100,
			initialSeatsExpected: 2,
		},
		{
			name:       "request verb is list, metadata.name specified, expect read 1.5MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?fieldSelector=metadata.name%3Dtest",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				Name:     "test",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 1_500_000},
			maxSeats:             100,
			initialSeatsExpected: 15,
		},
		{
			name:       "request verb is list, metadata.name, resource version 0, limit 500, expect read 200KB from cache",
			requestURI: "http://server/apis/foo.bar/v1/events?fieldSelector=metadata.name%3Dtest&limit=500&resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				Name:     "test",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 200_000},
			maxSeats:             100,
			initialSeatsExpected: 2,
		},
		{
			name:       "request verb is list, metadata.name, resource version 0, limit 500, expect seats capped by cache",
			requestURI: "http://server/apis/foo.bar/v1/events?fieldSelector=metadata.name%3Dtest&limit=500&resourceVersion=0",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				Name:     "test",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: maxObjectSize},
			maxSeats:             100,
			initialSeatsExpected: 10,
		},
		{
			name:       "request verb is list, labelSelector, expect read 8MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?labelSelector=app%3Dtest",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 80,
		},
		{
			name:       "request verb is list, labelSelector, limit 49, expect read 4MB from etcd by pagination",
			requestURI: "http://server/apis/foo.bar/v1/events?labelSelector=app%3Dtest&limit=49",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 40,
		},
		{
			name:       "request verb is list, labelSelector, limit 699, expect read 7MB from etcd",
			requestURI: "http://server/apis/foo.bar/v1/events?labelSelector=app%3Dtest&limit=699",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 70,
		},
		{
			name:       "request verb is list, labelSelector, resource version 0, expect seats capped cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0&labelSelector=app%3Dtest",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 10,
		},
		{
			name:       "request verb is list, labelSelector, resource version 0, limit 299, expect read 300KB from cache",
			requestURI: "http://server/apis/foo.bar/v1/events?resourceVersion=0&labelSelector=app%3Dtest&limit=29",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "list",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 3,
		},
		{
			name:       "request verb is watch, sendInitialEvents is nil",
			requestURI: "http://server/apis/foo.bar/v1/events?watch=true",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "watch",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is watch, sendInitialEvents is false",
			requestURI: "http://server/apis/foo.bar/v1/events?watch=true&sendInitialEvents=false",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "watch",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 10_000},
			maxSeats:             100,
			initialSeatsExpected: 1,
		},
		{
			name:       "request verb is watch, sendInitialEvents is true",
			requestURI: "http://server/apis/foo.bar/v1/events?watch=true&sendInitialEvents=true",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "watch",
				APIGroup: "foo.bar",
				Resource: "events",
			},
			stats:                storage.Stats{ObjectCount: 799, EstimatedAverageObjectSizeBytes: 1_000},
			maxSeats:             100,
			initialSeatsExpected: 8,
		},
		{
			name:       "request verb is create, no watches",
			requestURI: "http://server/apis/foo.bar/v1/foos",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "create",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        0,
			additionalLatencyExpected: 0,
		},
		{
			name:       "request verb is create, watches registered",
			requestURI: "http://server/apis/foo.bar/v1/foos",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "create",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			watchCount:                29,
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        3,
			additionalLatencyExpected: 5 * time.Millisecond,
		},
		{
			name:       "request verb is create, watches registered, no additional latency",
			requestURI: "http://server/apis/foo.bar/v1/foos",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "create",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			watchCount:                5,
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        0,
			additionalLatencyExpected: 0,
		},
		{
			name:       "request verb is create, watches registered, capped by watch cache",
			requestURI: "http://server/apis/foo.bar/v1/foos",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "create",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			watchCount:                199,
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        10,
			additionalLatencyExpected: 10 * time.Millisecond,
		},
		{
			name:       "request verb is update, no watches",
			requestURI: "http://server/apis/foo.bar/v1/foos/myfoo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "update",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        0,
			additionalLatencyExpected: 0,
		},
		{
			name:       "request verb is update, watches registered",
			requestURI: "http://server/apis/foor.bar/v1/foos/myfoo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "update",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			watchCount:                29,
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        3,
			additionalLatencyExpected: 5 * time.Millisecond,
		},
		{
			name:       "request verb is patch, no watches",
			requestURI: "http://server/apis/foo.bar/v1/foos/myfoo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "patch",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        0,
			additionalLatencyExpected: 0,
		},
		{
			name:       "request verb is patch, watches registered",
			requestURI: "http://server/apis/foo.bar/v1/foos/myfoo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "patch",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			watchCount:                29,
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        3,
			additionalLatencyExpected: 5 * time.Millisecond,
		},
		{
			name:       "request verb is patch, watches registered, lower max seats",
			requestURI: "http://server/apis/foo.bar/v1/foos/myfoo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "patch",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			watchCount:                100,
			maxSeats:                  5,
			initialSeatsExpected:      1,
			finalSeatsExpected:        5,
			additionalLatencyExpected: 10 * time.Millisecond,
		},
		{
			name:       "request verb is delete, no watches",
			requestURI: "http://server/apis/foo.bar/v1/foos/myfoo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "delete",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        0,
			additionalLatencyExpected: 0,
		},
		{
			name:       "request verb is delete, watches registered",
			requestURI: "http://server/apis/foo.bar/v1/foos/myfoo",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "delete",
				APIGroup: "foo.bar",
				Resource: "foos",
			},
			watchCount:                29,
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        3,
			additionalLatencyExpected: 5 * time.Millisecond,
		},
		{
			name:       "creating token for service account",
			requestURI: "http://server/api/v1/namespaces/foo/serviceaccounts/default/token",
			requestInfo: &apirequest.RequestInfo{
				Verb:        "create",
				APIGroup:    "v1",
				Resource:    "serviceaccounts",
				Subresource: "token",
			},
			watchCount:                5777,
			maxSeats:                  10,
			initialSeatsExpected:      1,
			finalSeatsExpected:        0,
			additionalLatencyExpected: 0,
		},
		{
			name:       "creating service account",
			requestURI: "http://server/api/v1/namespaces/foo/serviceaccounts",
			requestInfo: &apirequest.RequestInfo{
				Verb:     "create",
				APIGroup: "v1",
				Resource: "serviceaccounts",
			},
			watchCount:                1000,
			maxSeats:                  20,
			initialSeatsExpected:      1,
			finalSeatsExpected:        10,
			additionalLatencyExpected: 50 * time.Millisecond,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			statsFn := func(key string) (storage.Stats, error) {
				return test.stats, test.statsErr
			}
			watchCountsFn := func(_ *apirequest.RequestInfo) int {
				return test.watchCount
			}
			maxSeatsFn := func(_ string) uint64 {
				return test.maxSeats
			}

			estimator := NewWorkEstimator(statsFn, watchCountsFn, defaultCfg, maxSeatsFn)

			req, err := http.NewRequest("GET", test.requestURI, nil)
			if err != nil {
				t.Fatalf("Failed to create new HTTP request - %v", err)
			}

			if test.requestInfo != nil {
				req = req.WithContext(apirequest.WithRequestInfo(req.Context(), test.requestInfo))
			}

			workestimateGot := estimator.EstimateWork(req, "testFS", "testPL")
			if test.initialSeatsExpected != workestimateGot.InitialSeats {
				t.Errorf("Expected work estimate to match: %d initial seats, but got: %d", test.initialSeatsExpected, workestimateGot.InitialSeats)
			}
			if test.finalSeatsExpected != workestimateGot.FinalSeats {
				t.Errorf("Expected work estimate to match: %d final seats, but got: %d", test.finalSeatsExpected, workestimateGot.FinalSeats)
			}
			if test.additionalLatencyExpected != workestimateGot.AdditionalLatency {
				t.Errorf("Expected work estimate to match additional latency: %v, but got: %v", test.additionalLatencyExpected, workestimateGot.AdditionalLatency)
			}
		})
	}
}
