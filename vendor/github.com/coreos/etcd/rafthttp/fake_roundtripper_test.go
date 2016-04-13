// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.5

package rafthttp

import (
	"errors"
	"net/http"
)

func (t *roundTripperBlocker) RoundTrip(req *http.Request) (*http.Response, error) {
	c := make(chan struct{}, 1)
	t.mu.Lock()
	t.cancel[req] = c
	t.mu.Unlock()
	select {
	case <-t.unblockc:
		return &http.Response{StatusCode: http.StatusNoContent, Body: &nopReadCloser{}}, nil
	case <-req.Cancel:
		return nil, errors.New("request canceled")
	case <-c:
		return nil, errors.New("request canceled")
	}
}
