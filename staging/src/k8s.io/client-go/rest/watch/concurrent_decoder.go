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

//nolint:unused // This implementation is intentionally not wired into the watch path yet.
package watch

import (
	"fmt"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
)

type rawWatchEventResult struct {
	event *metav1.WatchEvent
	err   error
}

type bufferedWatchEventDecoder struct {
	streamDecoder streaming.Decoder
	rawResultsCh  chan rawWatchEventResult
	doneCh        chan struct{}
	closeOnce     sync.Once
}

func (d *bufferedWatchEventDecoder) close() {
	d.closeOnce.Do(func() {
		close(d.doneCh)
		if err := d.streamDecoder.Close(); err != nil {
			utilruntime.HandleError(fmt.Errorf("close watch event decoder: %w", err))
		}
	})
}

func (d *bufferedWatchEventDecoder) read() {
	defer close(d.rawResultsCh)
	for {
		select {
		case <-d.doneCh:
			return
		default:
		}

		result := d.readEvent()
		select {
		case <-d.doneCh:
			return
		case d.rawResultsCh <- result:
		}
		// Watch events are ordered, so there is no value in processing anything
		// that follows an error.
		if result.err != nil {
			return
		}
	}
}

func (d *bufferedWatchEventDecoder) readEvent() rawWatchEventResult {
	got, err := readEvent(d.streamDecoder)
	if err != nil {
		return rawWatchEventResult{err: err}
	}
	return rawWatchEventResult{event: &got}
}
