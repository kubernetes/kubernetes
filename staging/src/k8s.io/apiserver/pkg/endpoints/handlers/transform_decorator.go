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

package handlers

import (
	"context"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	utiltrace "k8s.io/utils/trace"
)

func transformResponseObjectWithTracker(ctx context.Context, scope *RequestScope, trace *utiltrace.Trace, req *http.Request, w http.ResponseWriter, statusCode int, mediaType negotiation.MediaTypeOptions, result runtime.Object) {
	startedAt := time.Now()
	defer func() {
		if tracker := request.ResponseWriteLatencyTrackerFrom(ctx); tracker != nil {
			count := 1
			if length := meta.LenList(result); length > 0 {
				count = length
			}
			tracker.TrackTransform(count, time.Since(startedAt))
		}
	}()

	transformResponseObject(ctx, scope, trace, req, w, statusCode, mediaType, result)
}
