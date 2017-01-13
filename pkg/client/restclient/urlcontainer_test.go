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

package restclient

import (
	"fmt"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

func TestURLContainerExclude(t *testing.T) {
	urls := make([]*url.URL, 0, 2)
	for i := 0; i < 2; i++ {
		u, _ := url.Parse(fmt.Sprintf("http://localhost:808%d", i))
		urls = append(urls, u)
	}
	container := NewURLContainer(urls)
	container.initializeRateLimiter = func(_ float32, _ int) flowcontrol.RateLimiter {
		return flowcontrol.NewFakeAlwaysRateLimiter()
	}
	container.Get()
	if container.stickyURL == nil {
		t.Errorf("After GET container should select some URL as valid")
	}
	container.Exclude(container.stickyURL)
	if container.stickyURL != nil {
		t.Errorf("After exclude container will invalidate currently selected URL")
	}
}
