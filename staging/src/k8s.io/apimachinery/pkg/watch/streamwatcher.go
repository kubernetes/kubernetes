/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io"
	"sync"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// Decoder allows StreamWatcher to watch any stream for which a Decoder can be written.
type Decoder interface {
	// Decode should return the type of event, the decoded object, or an error.
	// An error will cause StreamWatcher to call Close(). Decode should block until
	// it has data or an error occurs.
	Decode() (action EventType, object runtime.Object, err error)

	// Close should close the underlying io.Reader, signalling to the source of
	// the stream that it is no longer being watched. Close() must cause any
	// outstanding call to Decode() to return with an error of some sort.
	Close()
}

// Reporter hides the details of how an error is turned into a runtime.Object for
// reporting on a watch stream since this package may not import a higher level report.
type Reporter interface {
	// AsObject must convert err into a valid runtime.Object for the watch stream.
	AsObject(err error) runtime.Object
}

// StreamWatcher turns any stream for which you can write a Decoder interface
// into a watch.Interface.
type StreamWatcher struct {
	sync.Mutex
	source   Decoder
	reporter Reporter
	result   chan Event
	stopped  bool
}

// NewStreamWatcher creates a StreamWatcher from the given decoder.
func NewStreamWatcher(d Decoder, r Reporter) *StreamWatcher {
	sw := &StreamWatcher{
		source:   d,
		reporter: r,
		// It's easy for a consumer to add buffering via an extra
		// goroutine/channel, but impossible for them to remove it,
		// so nonbuffered is better.
		result: make(chan Event),
	}
	go sw.receive()
	return sw
}

// ResultChan implements Interface.
func (sw *StreamWatcher) ResultChan() <-chan Event {
	return sw.result
}

// Stop implements Interface.
func (sw *StreamWatcher) Stop() {
	// Call Close() exactly once by locking and setting a flag.
	sw.Lock()
	defer sw.Unlock()
	if !sw.stopped {
		sw.stopped = true
		sw.source.Close()
	}
}

// stopping returns true if Stop() was called previously.
func (sw *StreamWatcher) stopping() bool {
	sw.Lock()
	defer sw.Unlock()
	return sw.stopped
}

// receive reads result from the decoder in a loop and sends down the result channel.
func (sw *StreamWatcher) receive() {
	defer close(sw.result)
	defer sw.Stop()
	defer utilruntime.HandleCrash()
	for {
		action, obj, err := sw.source.Decode()
		if err != nil {
			// Ignore expected error.
			if sw.stopping() {
				return
			}
			switch err {
			case io.EOF:
				// watch closed normally
			case io.ErrUnexpectedEOF:
				klog.V(1).Infof("Unexpected EOF during watch stream event decoding: %v", err)
			default:
				if net.IsProbableEOF(err) || net.IsTimeout(err) {
					klog.V(5).Infof("Unable to decode an event from the watch stream: %v", err)
				} else {
					sw.result <- Event{
						Type:   Error,
						Object: sw.reporter.AsObject(fmt.Errorf("unable to decode an event from the watch stream: %v", err)),
					}
				}
			}
			return
		}
		sw.result <- Event{
			Type:   action,
			Object: obj,
		}
	}
}
