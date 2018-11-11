/*
Copyright 2017 The Kubernetes Authors.

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

package metrics

import (
	"strings"
	"testing"

	"k8s.io/client-go/util/flowcontrol"
)

func TestRegisterMetricAndTrackRateLimiterUsage(t *testing.T) {
	testCases := []struct {
		ownerName   string
		rateLimiter flowcontrol.RateLimiter
		err         string
	}{
		{
			ownerName:   "owner_name",
			rateLimiter: flowcontrol.NewTokenBucketRateLimiter(1, 1),
			err:         "",
		},
		{
			ownerName:   "owner_name",
			rateLimiter: flowcontrol.NewTokenBucketRateLimiter(1, 1),
			err:         "already registered",
		},
		{
			ownerName:   "invalid-owner-name",
			rateLimiter: flowcontrol.NewTokenBucketRateLimiter(1, 1),
			err:         "error registering rate limiter usage metric",
		},
	}

	for i, tc := range testCases {
		e := RegisterMetricAndTrackRateLimiterUsage(tc.ownerName, tc.rateLimiter)
		if e != nil {
			if tc.err == "" {
				t.Errorf("[%d] unexpected error: %v", i, e)
			} else if !strings.Contains(e.Error(), tc.err) {
				t.Errorf("[%d] expected an error containing %q: %v", i, tc.err, e)
			}
		}
	}
}
