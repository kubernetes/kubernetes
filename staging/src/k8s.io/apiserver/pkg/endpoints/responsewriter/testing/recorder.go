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
	"time"
)

// WithFakeResponseController extends a given httptest.ResponseRecorder object
// with a fake implementation of http.ResonseController.
// NOTE: use this function for testing purposes only.
//
// httptest.ResponseRecorder does not implement SetReadDeadline or
// SetWriteDeadline, see https://github.com/golang/go/issues/60229
// for more details.
func WithFakeResponseController(w *httptest.ResponseRecorder) *FakeResponseController {
	return &FakeResponseController{ResponseRecorder: w}
}

type Deadlines struct {
	Reads  []time.Time
	Writes []time.Time
}

type FakeResponseController struct {
	*httptest.ResponseRecorder

	// set the following errors respectively if you want SetReadDeadline
	// and/or SetWriteDeadline to return error.
	ReadDeadlineErr  error
	WriteDeadlineErr error

	// the user supplied deadline to SetReadDeadline and SetWriteDeadline
	// are stored in these slices respectively, this enables us to verify
	// whether the expected deadline was passed to the controller.
	readDeadlines  []time.Time
	writeDeadlines []time.Time
}

func (w *FakeResponseController) SetReadDeadline(deadline time.Time) error {
	// we add to the slice first as a proof of invocation
	w.readDeadlines = append(w.readDeadlines, deadline)
	return w.ReadDeadlineErr
}

func (w *FakeResponseController) SetWriteDeadline(deadline time.Time) error {
	// we add to the slice first as a proof of invocation
	w.writeDeadlines = append(w.writeDeadlines, deadline)
	return w.WriteDeadlineErr
}

func (w *FakeResponseController) GetDeadlines() Deadlines {
	return Deadlines{
		Reads:  w.readDeadlines,
		Writes: w.writeDeadlines,
	}
}
