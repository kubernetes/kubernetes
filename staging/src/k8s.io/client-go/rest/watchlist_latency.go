/*
Copyright The Kubernetes Authors.

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
	"context"
	"net/url"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/metrics"
)

// Measure completion in stream order so both decoder implementations include all
// initial objects. Caller backpressure can delay decoding and increase the latency.
type watchListLatencyRecordingDecoder struct {
	watch.Decoder
	ctx   context.Context
	verb  string
	url   url.URL
	start time.Time
}

func (d *watchListLatencyRecordingDecoder) Decode() (watch.EventType, runtime.Object, error) {
	eventType, obj, err := d.Decoder.Decode()
	if err == nil && eventType == watch.Bookmark && !d.start.IsZero() {
		if accessor, err := meta.Accessor(obj); err == nil && accessor.GetAnnotations()[metav1.InitialEventsAnnotationKey] == "true" {
			metrics.WatchListLatency.Observe(d.ctx, d.verb, d.url, time.Since(d.start))
			d.start = time.Time{}
		}
	}
	return eventType, obj, err
}
