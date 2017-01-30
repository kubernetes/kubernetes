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

package rest

import (
	"fmt"
	"net/url"
	"testing"

	"k8s.io/client-go/util/flowcontrol"
)

func TestStickyURLContainer(t *testing.T) {
	urls := make([]*url.URL, 0, 2)
	for i := 0; i < 2; i++ {
		u, _ := url.Parse(fmt.Sprintf("http://localhost:808%d", i))
		urls = append(urls, u)
	}
	container := &URLContainer{
		order: urls,
		initializeRateLimiter: func(_ float32, _ int) flowcontrol.RateLimiter {
			return flowcontrol.NewFakeNeverRateLimiter()
		},
	}
	container.renewRateLimiter()
	container.renewStickyURL()
	firstURL := container.Get()
	container.Exclude(firstURL)
	secondURL := container.Get()
	if secondURL == firstURL {
		t.Errorf("After first exclude container should change URL from first one")
	}
	if gotURL := container.Get(); gotURL != secondURL {
		t.Errorf("After first exclude container should use second URL as valid, but got %v", gotURL)
	}
	container.Exclude(secondURL)
	if gotURL := container.Get(); gotURL != firstURL {
		t.Errorf("After second exclude container should return back to first URL, but got %v", gotURL)
	}
}
