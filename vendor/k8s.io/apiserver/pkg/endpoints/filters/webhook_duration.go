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

package filters

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
)

var (
	watchVerbs = sets.NewString("watch")
)

// WithLatencyTrackers adds a LatencyTrackers instance to the
// context associated with a request so that we can measure latency
// incurred in various components within the apiserver.
func WithLatencyTrackers(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			handleError(w, req, http.StatusInternalServerError, fmt.Errorf("no RequestInfo found in context, handler chain must be wrong"))
			return
		}

		if watchVerbs.Has(requestInfo.Verb) {
			handler.ServeHTTP(w, req)
			return
		}

		req = req.WithContext(request.WithLatencyTrackers(ctx))
		w = responsewriter.WrapForHTTP1Or2(&writeLatencyTracker{
			ResponseWriter: w,
			ctx:            req.Context(),
		})

		handler.ServeHTTP(w, req)
	})
}

var _ http.ResponseWriter = &writeLatencyTracker{}
var _ responsewriter.UserProvidedDecorator = &writeLatencyTracker{}

type writeLatencyTracker struct {
	http.ResponseWriter
	ctx context.Context
}

func (wt *writeLatencyTracker) Unwrap() http.ResponseWriter {
	return wt.ResponseWriter
}

func (wt *writeLatencyTracker) Write(bs []byte) (int, error) {
	startedAt := time.Now()
	defer func() {
		request.TrackResponseWriteLatency(wt.ctx, time.Since(startedAt))
	}()

	return wt.ResponseWriter.Write(bs)
}
