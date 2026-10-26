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

package watch

import (
	"io"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/framer"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestBufferedWatchEventDecoderCloseWithFullQueue(t *testing.T) {
	stream := &observedWatchEventDecoder{
		Decoder: streaming.NewDecoder(framer.NewJSONFramedReader(io.NopCloser(strings.NewReader(`{"type":"ADDED"}
{"type":"MODIFIED"}`))), getDecoder()),
		decoded: make(chan struct{}, 2),
	}
	decoder := &bufferedWatchEventDecoder{
		streamDecoder: stream,
		rawResultsCh:  make(chan rawWatchEventResult, 1),
		doneCh:        make(chan struct{}),
	}
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		decoder.read()
	}()

	// The second decode proves that the first event was queued and the next send cannot complete.
	<-stream.decoded
	<-stream.decoded
	decoder.close()
	select {
	case <-exited:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("decoder did not exit after close with a full result queue")
	}
}

type observedWatchEventDecoder struct {
	streaming.Decoder
	decoded chan struct{}
}

func (d *observedWatchEventDecoder) Decode(defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	obj, gvk, err := d.Decoder.Decode(defaults, into)
	if err == nil {
		d.decoded <- struct{}{}
	}
	return obj, gvk, err
}
