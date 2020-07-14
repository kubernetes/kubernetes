/*
Copyright 2019 The Kubernetes Authors.

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

package internal

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLazyTruncatedUserAgent(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}

	ua := "short-agent"
	req.Header.Set("User-Agent", ua)
	uaNotTruncated := &LazyTruncatedUserAgent{req}
	assert.Equal(t, ua, fmt.Sprintf("%v", uaNotTruncated))

	ua = ""
	for i := 0; i < maxUserAgentLength*2; i++ {
		ua = ua + "a"
	}
	req.Header.Set("User-Agent", ua)
	uaTruncated := &LazyTruncatedUserAgent{req}
	assert.NotEqual(t, ua, fmt.Sprintf("%v", uaTruncated))

	usUnknown := &LazyTruncatedUserAgent{}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", usUnknown))
}

func TestLazyClientIP(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}

	ip := "127.0.0.1"
	req.Header.Set("X-Forwarded-For", ip)

	clientIPWithReq := &LazyClientIP{req}
	assert.Equal(t, ip, fmt.Sprintf("%v", clientIPWithReq))

	clientIPWithoutReq := &LazyClientIP{}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", clientIPWithoutReq))
}
