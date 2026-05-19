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
	"context"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
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
	compbasemetrics "k8s.io/component-base/metrics"
)

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
		// nothing will ever be sent down this channel
		return nil, func() bool { return false }
	}
	t := time.NewTimer(w.timeout)
	return t.C, t.Stop
}

// serveWatchHandler returns a handle to serve a watch response.
// TODO: the functionality in this method and in WatchServer.Serve is not cleanly decoupled.
func serveWatchHandler(watcher watch.Interface, scope *RequestScope, mediaTypeOptions negotiation.MediaTypeOptions, req *http.Request, w http.ResponseWriter, timeout time.Duration, metricsScope string) (http.Handler, error) {
	options, err := optionsForTransform(mediaTypeOptions, req)
	if err != nil {
		return nil, err
	}

	// negotiate for the stream serializer from the scope's serializer
	serializer, err := negotiation.NegotiateOutputMediaTypeStream(req, scope.Serializer, scope)
	if err != nil {
		return nil, err
	}
	framer := serializer.StreamSerializer.Framer
	var encoder runtime.Encoder
	if utilfeature.DefaultFeatureGate.Enabled(features.CBORServingAndStorage) {
		encoder = scope.Serializer.EncoderForVersion(runtime.UseNondeterministicEncoding(serializer.StreamSerializer.Serializer), scope.Kind.GroupVersion())
	} else {
		encoder = scope.Serializer.EncoderForVersion(serializer.StreamSerializer.Serializer, scope.Kind.GroupVersion())
	}
	useTextFraming := serializer.EncodesAsText
	if framer == nil {
		return nil, fmt.Errorf("no framer defined for %q available for embedded encoding", serializer.MediaType)
	}
	// TODO: next step, get back mediaTypeOptions from negotiate and return the exact value here
	mediaType := serializer.MediaType
	switch mediaType {
	case runtime.ContentTypeJSON:
		// as-is
	case runtime.ContentTypeCBOR:
		// If a client indicated it accepts application/cbor (exactly one data item) on a
		// watch request, set the conformant application/cbor-seq media type the watch
		// response. RFC 9110 allows an origin server to deviate from the indicated
		// preference rather than send a 406 (Not Acceptable) response (see
		// https://www.rfc-editor.org/rfc/rfc9110.html#section-12.1-5).
		mediaType = runtime.ContentTypeCBORSequence
	default:
		mediaType += ";stream=watch"
	}

	ctx := req.Context()

	// locate the appropriate embedded encoder based on the transform
	var negotiatedEncoder runtime.Encoder
	contentKind, contentSerializer, transform := targetEncodingForTransform(scope, mediaTypeOptions, req)
	if transform {
		info, ok := runtime.SerializerInfoForMediaType(contentSerializer.SupportedMediaTypes(), serializer.MediaType)
		if !ok {
			return nil, fmt.Errorf("no encoder for %q exists in the requested target %#v", serializer.MediaType, contentSerializer)
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.CBORServingAndStorage) {
			negotiatedEncoder = contentSerializer.EncoderForVersion(runtime.UseNondeterministicEncoding(info.Serializer), contentKind.GroupVersion())
		} else {
			negotiatedEncoder = contentSerializer.EncoderForVersion(info.Serializer, contentKind.GroupVersion())
		}
	} else {
		if utilfeature.DefaultFeatureGate.Enabled(features.CBORServingAndStorage) {
			negotiatedEncoder = scope.Serializer.EncoderForVersion(runtime.UseNondeterministicEncoding(serializer.Serializer), contentKind.GroupVersion())
		} else {
			negotiatedEncoder = scope.Serializer.EncoderForVersion(serializer.Serializer, contentKind.GroupVersion())
		}
	}

	var memoryAllocator runtime.MemoryAllocator

	if encoderWithAllocator, supportsAllocator := negotiatedEncoder.(runtime.EncoderWithAllocator); supportsAllocator {
		// don't put the allocator inside the embeddedEncodeFn as that would allocate memory on every call.
		// instead, we allocate the buffer for the entire watch session and release it when we close the connection.
		memoryAllocator = runtime.AllocatorPool.Get().(*runtime.Allocator)
		negotiatedEncoder = runtime.NewEncoderWithAllocator(encoderWithAllocator, memoryAllocator)
	}
	var tableOptions *metav1.TableOptions
	if options != nil {
		if passedOptions, ok := options.(*metav1.TableOptions); ok {
			tableOptions = passedOptions
		} else {
			return nil, fmt.Errorf("unexpected options type: %T", options)
		}
	}
	embeddedEncoder := newWatchEmbeddedEncoder(ctx, negotiatedEncoder, mediaTypeOptions.Convert, tableOptions, scope)

	if encoderWithAllocator, supportsAllocator := encoder.(runtime.EncoderWithAllocator); supportsAllocator {
		if memoryAllocator == nil {
			// don't put the allocator inside the embeddedEncodeFn as that would allocate memory on every call.
			// instead, we allocate the buffer for the entire watch session and release it when we close the connection.
			memoryAllocator = runtime.AllocatorPool.Get().(*runtime.Allocator)
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

		MemoryAllocator:      memoryAllocator,
		TimeoutFactory:       &realTimeoutFactory{timeout},
		ServerShuttingDownCh: serverShuttingDownCh,

		metricsScope: metricsScope,
	}

	if wsstream.IsWebSocketRequest(req) {
		w.Header().Set("Content-Type", server.MediaType)
		return websocket.Handler(server.HandleWS), nil
	}
	return http.HandlerFunc(server.HandleHTTP), nil
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

	MemoryAllocator      runtime.MemoryAllocator
	TimeoutFactory       TimeoutFactory
	ServerShuttingDownCh <-chan struct{}

	metricsScope string
}

// watchEventMetricsRecorder allows the caller to count bytes written and report the size of the event.
// It is thread-safe, as long as underlying io.Writer is thread-safe.
// Once all Write calls for a given watch event have finished, RecordEvent must be called.
type watchEventMetricsRecorder struct {
	writer      io.Writer
	countMetric compbasemetrics.CounterMetric
	sizeMetric  compbasemetrics.ObserverMetric
	byteCount   atomic.Int64
}

// Write implements io.Writer.
func (c *watchEventMetricsRecorder) Write(p []byte) (n int, err error) {
	n, err = c.writer.Write(p)
	c.byteCount.Add(int64(n))
	return
}

// Record reports the metrics and resets the byte count.
func (c *watchEventMetricsRecorder) RecordEvent() {
	c.countMetric.Inc()
	c.sizeMetric.Observe(float64(c.byteCount.Swap(0)))
}

// HandleHTTP serves a series of encoded events via HTTP with Transfer-Encoding: chunked.
// or over a websocket connection.
func (s *WatchServer) HandleHTTP(w http.ResponseWriter, req *http.Request) {
	defer func() {
		if s.MemoryAllocator != nil {
			runtime.AllocatorPool.Put(s.MemoryAllocator)
		}
	}()

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

	gvr := s.Scope.Resource

	recorder := &watchEventMetricsRecorder{
		writer:      framer,
		countMetric: metrics.WatchEvents.WithContext(req.Context()).WithLabelValues(gvr.Group, gvr.Version, gvr.Resource),
		sizeMetric:  metrics.WatchEventsSizes.WithContext(req.Context()).WithLabelValues(gvr.Group, gvr.Version, gvr.Resource),
	}

	watchEncoder := newWatchEncoder(req.Context(), gvr, s.EmbeddedEncoder, s.Encoder, recorder)
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
			isWatchListLatencyRecordingRequired := shouldRecordWatchListLatency(event)

			if err := watchEncoder.Encode(event); err != nil {
				utilruntime.HandleError(err)
				// client disconnect.
				return
			}
			recorder.RecordEvent()

			if len(ch) == 0 {
				flusher.Flush()
			}
			if isWatchListLatencyRecordingRequired {
				metrics.RecordWatchListLatency(req.Context(), s.Scope.Resource, s.metricsScope)
			}
		}
	}
}

// HandleWS serves a series of encoded events over a websocket connection.
func (s *WatchServer) HandleWS(ws *websocket.Conn) {
	defer func() {
		if s.MemoryAllocator != nil {
			runtime.AllocatorPool.Put(s.MemoryAllocator)
		}
	}()

	defer ws.Close()
	done := make(chan struct{})
	// ensure the connection times out
	timeoutCh, cleanup := s.TimeoutFactory.TimeoutCh()
	defer cleanup()

	go func() {
		defer utilruntime.HandleCrash()
		// This blocks until the connection is closed.
		// Client should not send anything.
		wsstream.IgnoreReceives(ws, 0)
		// Once the client closes, we should also close
		close(done)
	}()

	framer := newWebsocketFramer(ws, s.UseTextFraming)

	gvr := s.Scope.Resource
	watchEncoder := newWatchEncoder(context.TODO(), gvr, s.EmbeddedEncoder, s.Encoder, framer)
	ch := s.Watching.ResultChan()

	for {
		select {
		case <-done:
			return
		case <-timeoutCh:
			return
		case event, ok := <-ch:
			if !ok {
				// End of results.
				return
			}

			if err := watchEncoder.Encode(event); err != nil {
				utilruntime.HandleError(err)
				// client disconnect.
				return
			}
		}
	}
}

type websocketFramer struct {
	ws             *websocket.Conn
	useTextFraming bool
}

func newWebsocketFramer(ws *websocket.Conn, useTextFraming bool) io.Writer {
	return &websocketFramer{
		ws:             ws,
		useTextFraming: useTextFraming,
	}
}

func (w *websocketFramer) Write(p []byte) (int, error) {
	if w.useTextFraming {
		// bytes.Buffer::String() has a special handling of nil value, but given
		// we're writing serialized watch events, this will never happen here.
		if err := websocket.Message.Send(w.ws, string(p)); err != nil {
			return 0, err
		}
		return len(p), nil
	}
	if err := websocket.Message.Send(w.ws, p); err != nil {
		return 0, err
	}
	return len(p), nil
}

var _ io.Writer = &websocketFramer{}

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
