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

package client

import (
	"errors"
	"net/http"
)

func (t *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	select {
	case resp := <-t.respchan:
		return resp, nil
	case err := <-t.errchan:
		return nil, err
	case <-t.startCancel:
	case <-req.Cancel:
	}
	select {
	// this simulates that the request is finished before cancel effects
	case resp := <-t.respchan:
		return resp, nil
	// wait on finishCancel to simulate taking some amount of
	// time while calling CancelRequest
	case <-t.finishCancel:
		return nil, errors.New("cancelled")
	}
}
