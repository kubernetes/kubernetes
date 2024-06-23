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

package filters

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
)

func TestWithPerRequestDeadline(t *testing.T) {
	tests := []struct {
		name             string
		ctx              func() (context.Context, context.CancelFunc)
		readDeadlineErr  error
		writeDeadlineErr error
		invoked          int
		statusCode       int
	}{
		{
			name: "deadline not set, read/write deadline should not be in effect",
			ctx: func() (context.Context, context.CancelFunc) {
				return context.WithCancel(context.Background())
			},
			invoked:    1,
			statusCode: http.StatusOK,
		},
		{
			name: "deadline set, read/write deadline should be in effect",
			ctx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Minute)
			},
			invoked:    1,
			statusCode: http.StatusOK,
		},
		{
			name: "deadline set, error while setting write deadline, abort request",
			ctx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Minute)
			},
			writeDeadlineErr: errors.New("foo error"),
			invoked:          0,
			statusCode:       http.StatusInternalServerError,
		},
		{
			name: "deadline set, error while setting read deadline, abort request",
			ctx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Minute)
			},
			readDeadlineErr: errors.New("bar error"),
			invoked:         0,
			statusCode:      http.StatusInternalServerError,
		},
		{
			name: "deadline set, error while setting read/write deadline, abort request",
			ctx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Minute)
			},
			writeDeadlineErr: errors.New("foo error"),
			readDeadlineErr:  errors.New("bar error"),
			invoked:          0,
			statusCode:       http.StatusInternalServerError,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var invoked int
			handler := WithPerRequestDeadline(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				invoked++
			}))

			req, err := http.NewRequest(http.MethodGet, "/ping", nil)
			if err != nil {
				t.Fatalf("failed to create a new http request - %v", err)
			}

			// attach the desired context to the request
			ctx, cancel := test.ctx()
			defer cancel()
			req = req.WithContext(ctx)

			w := &responsewritertesting.FakeResponseController{
				ResponseRecorder: httptest.NewRecorder(),
				ReadDeadlineErr:  test.readDeadlineErr,
				WriteDeadlineErr: test.writeDeadlineErr,
			}

			handler.ServeHTTP(w, req)

			result := w.GetDeadlines()
			deadline, ok := ctx.Deadline()
			switch {
			case ok:
				// when deadline is set, we always expect SetWriteDeadline to be invoked
				if want, got := []time.Time{deadline}, result.Writes; !cmp.Equal(want, got) {
					t.Errorf("expected write deadline to be set - diff: %s", cmp.Diff(want, got))
				}

				switch {
				// SetReadDeadline should not be invoked, when we expect SetWriteDeadline to return an error
				case test.writeDeadlineErr != nil:
					if len(result.Reads) != 0 {
						t.Errorf("did not expect read deadline to be set: %v", result.Reads)
					}
				// we expect an attempt to set read deadline, when SetWriteDeadline is successful
				default:
					if want, got := []time.Time{deadline}, result.Reads; !cmp.Equal(want, got) {
						t.Errorf("expected read deadline to be set - diff: %s", cmp.Diff(want, got))
					}
				}
			default:
				// deadline not set in the context, so neither read not write deadline should be set
				if len(result.Reads) != 0 {
					t.Errorf("did not expect read deadline to be set: %v", result.Reads)
				}
				if len(result.Writes) != 0 {
					t.Errorf("did not expect write deadline to be set: %v", result.Writes)
				}
			}

			if want, got := test.invoked, invoked; want != got {
				t.Errorf("expected the handler to be invoked %d times, but got: %d", want, got)
			}
			if want, got := test.statusCode, w.Result().StatusCode; want != got {
				t.Errorf("expected HTTP status code: %d, but got: %d", want, got)
			}
		})
	}
}
