/*
Copyright 2024 The Kubernetes Authors.

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

package testing

import (
	"net/http/httptest"
	"strings"
	"time"
)

// WithFakeResponseController extends a given httptest.ResponseRecorder object
// with a fake implementation of http.ResonseController.
// NOTE: use this function for testing purposes only.
//
// httptest.ResponseRecorder does not implement SetReadDeadline or
// SetWriteDeadline, see https://github.com/golang/go/issues/60229
// for more details.
// TODO: once https://github.com/golang/go/issues/60229 is fixed
// we can remove this function.
func WithFakeResponseController(w *httptest.ResponseRecorder) *FakeResponseRecorder {
	return &FakeResponseRecorder{ResponseRecorder: w}
}

type FakeResponseRecorder struct {
	*httptest.ResponseRecorder

	ReadDeadlines  []time.Time
	WriteDeadlines []time.Time
}

func (w *FakeResponseRecorder) SetReadDeadline(deadline time.Time) error {
	w.ReadDeadlines = append(w.ReadDeadlines, deadline)
	return nil
}

func (w *FakeResponseRecorder) SetWriteDeadline(deadline time.Time) error {
	w.WriteDeadlines = append(w.WriteDeadlines, deadline)
	return nil
}

func IsStreamReadOrWriteTimeout(err error) bool {
	if err == nil {
		return false
	}
	// we get the the following stream reset error due to a
	// timeout from the per handler write timeout.
	return strings.Contains(err.Error(), "stream error: stream ID") &&
		strings.Contains(err.Error(), "INTERNAL_ERROR")
}
