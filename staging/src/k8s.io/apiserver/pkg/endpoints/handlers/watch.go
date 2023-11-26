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

package handlers

import (
	"bytes"
	"fmt"
	"net/http"
	"time"

	"golang.org/x/net/websocket"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// nothing will ever be sent down this channel
var neverExitWatch <-chan time.Time = make(chan time.Time)

// timeoutFactory abstracts watch timeout logic for testing
type TimeoutFactory interface {
	TimeoutCh() (<-chan time.Time, func() bool)
}

// realTimeoutFactory implements timeoutFactory
type realTimeoutFactory struct {
	timeout time.Duration
}

// TimeoutCh returns a channel which will receive something when the watch times out,
// and a cleanup function to call when this happens.
func (w *realTimeoutFactory) TimeoutCh() (<-chan time.Time, func() bool) {
	if w.timeout == 0 {
		return neverExitWatch, func() bool { return false }
	}
	t := time.NewTimer(w.timeout)
	return t.C, t.Stop
}

// serveWatch will serve a watch response.
// TODO: the functionality in this method and in WatchServer.Serve is not cleanly decoupled.
func serveWatch(watcher watch.Interface, scope *RequestScope, mediaTypeOptions negotiation.MediaTypeOptions, req *http.Request, w http.ResponseWriter, timeout time.Duration, metricsScope string) {
	defer watcher.Stop()

	options, err := optionsForTransform(mediaTypeOptions, req)
	if err != nil {
		scope.err(err, w, req)
		return
	}

	// negotiate for the stream serializer from the scope's serializer
	serializer, err := negotiation.NegotiateOutputMediaTypeStream(req, scope.Serializer, scope)
	if err != nil {
		scope.err(err, w, req)
		return
	}
	framer := serializer.StreamSerializer.Framer
	streamSerializer := serializer.StreamSerializer.Serializer
	encoder := scope.Serializer.EncoderForVersion(streamSerializer, scope.Kind.GroupVersion())
	useTextFraming := serializer.EncodesAsText
	if framer == nil {
		scope.err(fmt.Errorf("no framer defined for %q available for embedded encoding", serializer.MediaType), w, req)
		return
	}
	// TODO: next step, get back mediaTypeOptions from negotiate and return the exact value here
	mediaType := serializer.MediaType
	if mediaType != runtime.ContentTypeJSON {
		mediaType += ";stream=watch"
	}

	ctx := req.Context()

	// locate the appropriate embedded encoder based on the transform
	var embeddedEncoder runtime.Encoder
	contentKind, contentSerializer, transform := targetEncodingForTransform(scope, mediaTypeOptions, req)
	if transform {
		info, ok := runtime.SerializerInfoForMediaType(contentSerializer.SupportedMediaTypes(), serializer.MediaType)
		if !ok {
			scope.err(fmt.Errorf("no encoder for %q exists in the requested target %#v", serializer.MediaType, contentSerializer), w, req)
			return
		}
		embeddedEncoder = contentSerializer.EncoderForVersion(info.Serializer, contentKind.GroupVersion())
	} else {
		embeddedEncoder = scope.Serializer.EncoderForVersion(serializer.Serializer, contentKind.GroupVersion())
	}

	var memoryAllocator runtime.MemoryAllocator

	if encoderWithAllocator, supportsAllocator := embeddedEncoder.(runtime.EncoderWithAllocator); supportsAllocator {
		// don't put the allocator inside the embeddedEncodeFn as that would allocate memory on every call.
		// instead, we allocate the buffer for the entire watch session and release it when we close the connection.
		memoryAllocator = runtime.AllocatorPool.Get().(*runtime.Allocator)
		defer runtime.AllocatorPool.Put(memoryAllocator)
		embeddedEncoder = runtime.NewEncoderWithAllocator(encoderWithAllocator, memoryAllocator)
	}
	var tableOptions *metav1.TableOptions
	if options != nil {
		if passedOptions, ok := options.(*metav1.TableOptions); ok {
			tableOptions = passedOptions
		} else {
			scope.err(fmt.Errorf("unexpected options type: %T", options), w, req)
			return
		}
	}
	embeddedEncoder = newWatchEmbeddedEncoder(ctx, embeddedEncoder, mediaTypeOptions.Convert, tableOptions, scope)

	if encoderWithAllocator, supportsAllocator := encoder.(runtime.EncoderWithAllocator); supportsAllocator {
		if memoryAllocator == nil {
			// don't put the allocator inside the embeddedEncodeFn as that would allocate memory on every call.
			// instead, we allocate the buffer for the entire watch session and release it when we close the connection.
			memoryAllocator = runtime.AllocatorPool.Get().(*runtime.Allocator)
			defer runtime.AllocatorPool.Put(memoryAllocator)
		}
		encoder = runtime.NewEncoderWithAllocator(encoderWithAllocator, memoryAllocator)
	}

	var serverShuttingDownCh <-chan struct{}
	if signals := apirequest.ServerShutdownSignalFrom(req.Context()); signals != nil {
		serverShuttingDownCh = signals.ShuttingDown()
	}

	server := &WatchServer{
		Watching: watcher,
		Scope:    scope,

		UseTextFraming:  useTextFraming,
		MediaType:       mediaType,
		Framer:          framer,
		Encoder:         encoder,
		EmbeddedEncoder: embeddedEncoder,

		TimeoutFactory:       &realTimeoutFactory{timeout},
		ServerShuttingDownCh: serverShuttingDownCh,

		metricsScope: metricsScope,
	}

	server.ServeHTTP(w, req)
}

// WatchServer serves a watch.Interface over a websocket or vanilla HTTP.
type WatchServer struct {
	Watching watch.Interface
	Scope    *RequestScope

	// true if websocket messages should use text framing (as opposed to binary framing)
	UseTextFraming bool
	// the media type this watch is being served with
	MediaType string
	// used to frame the watch stream
	Framer runtime.Framer
	// used to encode the watch stream event itself
	Encoder runtime.Encoder
	// used to encode the nested object in the watch stream
	EmbeddedEncoder runtime.Encoder

	TimeoutFactory       TimeoutFactory
	ServerShuttingDownCh <-chan struct{}

	metricsScope string
}

// ServeHTTP serves a series of encoded events via HTTP with Transfer-Encoding: chunked
// or over a websocket connection.
func (s *WatchServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	kind := s.Scope.Kind

	if wsstream.IsWebSocketRequest(req) {
		w.Header().Set("Content-Type", s.MediaType)
		websocket.Handler(s.HandleWS).ServeHTTP(w, req)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		err := fmt.Errorf("unable to start watch - can't get http.Flusher: %#v", w)
		utilruntime.HandleError(err)
		s.Scope.err(errors.NewInternalError(err), w, req)
		return
	}

	framer := s.Framer.NewFrameWriter(w)
	if framer == nil {
		// programmer error
		err := fmt.Errorf("no stream framing support is available for media type %q", s.MediaType)
		utilruntime.HandleError(err)
		s.Scope.err(errors.NewBadRequest(err.Error()), w, req)
		return
	}

	// ensure the connection times out
	timeoutCh, cleanup := s.TimeoutFactory.TimeoutCh()
	defer cleanup()

	// begin the stream
	w.Header().Set("Content-Type", s.MediaType)
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	watchEncoder := newWatchEncoder(req.Context(), kind, s.EmbeddedEncoder, s.Encoder, framer)
	ch := s.Watching.ResultChan()
	done := req.Context().Done()

	for {
		select {
		case <-s.ServerShuttingDownCh:
			// the server has signaled that it is shutting down (not accepting
			// any new request), all active watch request(s) should return
			// immediately here. The WithWatchTerminationDuringShutdown server
			// filter will ensure that the response to the client is rate
			// limited in order to avoid any thundering herd issue when the
			// client(s) try to reestablish the WATCH on the other
			// available apiserver instance(s).
			return
		case <-done:
			return
		case <-timeoutCh:
			return
		case event, ok := <-ch:
			if !ok {
				// End of results.
				return
			}
			metrics.WatchEvents.WithContext(req.Context()).WithLabelValues(kind.Group, kind.Version, kind.Kind).Inc()
			isWatchListLatencyRecordingRequired := shouldRecordWatchListLatency(event)

			if err := watchEncoder.Encode(event); err != nil {
				utilruntime.HandleError(err)
				// client disconnect.
				return
			}

			if len(ch) == 0 {
				flusher.Flush()
			}
			if isWatchListLatencyRecordingRequired {
				metrics.RecordWatchListLatency(req.Context(), s.Scope.Resource, s.metricsScope)
			}
		}
	}
}

// HandleWS implements a websocket handler.
func (s *WatchServer) HandleWS(ws *websocket.Conn) {
	defer ws.Close()
	done := make(chan struct{})

	go func() {
		defer utilruntime.HandleCrash()
		// This blocks until the connection is closed.
		// Client should not send anything.
		wsstream.IgnoreReceives(ws, 0)
		// Once the client closes, we should also close
		close(done)
	}()

	var unknown runtime.Unknown
	internalEvent := &metav1.InternalEvent{}
	buf := &bytes.Buffer{}
	streamBuf := &bytes.Buffer{}
	ch := s.Watching.ResultChan()

	for {
		select {
		case <-done:
			return
		case event, ok := <-ch:
			if !ok {
				// End of results.
				return
			}

			if err := s.EmbeddedEncoder.Encode(event.Object, buf); err != nil {
				// unexpected error
				utilruntime.HandleError(fmt.Errorf("unable to encode watch object %T: %v", event.Object, err))
				return
			}

			// ContentType is not required here because we are defaulting to the serializer
			// type
			unknown.Raw = buf.Bytes()
			event.Object = &unknown

			// the internal event will be versioned by the encoder
			// create the external type directly and encode it.  Clients will only recognize the serialization we provide.
			// The internal event is being reused, not reallocated so its just a few extra assignments to do it this way
			// and we get the benefit of using conversion functions which already have to stay in sync
			outEvent := &metav1.WatchEvent{}
			*internalEvent = metav1.InternalEvent(event)
			err := metav1.Convert_v1_InternalEvent_To_v1_WatchEvent(internalEvent, outEvent, nil)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to convert watch object: %v", err))
				// client disconnect.
				return
			}
			if err := s.Encoder.Encode(outEvent, streamBuf); err != nil {
				// encoding error
				utilruntime.HandleError(fmt.Errorf("unable to encode event: %v", err))
				return
			}
			if s.UseTextFraming {
				if err := websocket.Message.Send(ws, streamBuf.String()); err != nil {
					// Client disconnect.
					return
				}
			} else {
				if err := websocket.Message.Send(ws, streamBuf.Bytes()); err != nil {
					// Client disconnect.
					return
				}
			}
			buf.Reset()
			streamBuf.Reset()
		}
	}
}

func shouldRecordWatchListLatency(event watch.Event) bool {
	if event.Type != watch.Bookmark || !utilfeature.DefaultFeatureGate.Enabled(features.WatchList) {
		return false
	}
	// as of today the initial-events-end annotation is added only to a single event
	// by the watch cache and only when certain conditions are met
	//
	// for more please read https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/3157-watch-list
	hasAnnotation, err := storage.HasInitialEventsEndBookmarkAnnotation(event.Object)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to determine if the obj has the required annotation for measuring watchlist latency, obj %T: %v", event.Object, err))
		return false
	}
	return hasAnnotation
}
