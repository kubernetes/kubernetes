/*
Copyright 2016 The Kubernetes Authors.

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
	"bufio"
	"errors"
	"net"
	"net/http"
	"sync"

	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// WithAudit decorates a http.Handler with audit logging information for all the
// requests coming to the server. If out is nil, no decoration takes place.
// Each audit log contains two entries:
// 1. the request line containing:
//    - unique id allowing to match the response line (see 2)
//    - source ip of the request
//    - HTTP method being invoked
//    - original user invoking the operation
//    - original user's groups info
//    - impersonated user for the operation
//    - impersonated groups info
//    - namespace of the request or <none>
//    - uri is the full URI as requested
// 2. the response line containing:
//    - the unique id from 1
//    - response code
func WithAudit(handler http.Handler, requestContextMapper request.RequestContextMapper, sink audit.Sink, policy *auditinternal.Policy, longRunningCheck request.LongRunningRequestCheck) http.Handler {
	if sink == nil {
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, ok := requestContextMapper.Get(req)
		if !ok {
			responsewriters.InternalError(w, req, errors.New("no context found for request"))
			return
		}

		attribs, err := GetAuthorizerAttributes(ctx)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to GetAuthorizerAttributes: %v", err))
			responsewriters.InternalError(w, req, errors.New("failed to parse request"))
			return
		}

		ev, err := audit.NewEventFromRequest(req, policy, attribs)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to complete audit event from request: %v", err))
			responsewriters.InternalError(w, req, errors.New("failed to update context"))
			return
		}

		ctx = request.WithAuditEvent(ctx, ev)
		if err := requestContextMapper.Update(req, ctx); err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to attach audit event to the context: %v", err))
			responsewriters.InternalError(w, req, errors.New("failed to update context"))
			return
		}

		// intercept the status code
		longRunning := false
		var longRunningSink audit.Sink
		if longRunningCheck != nil {
			ri, _ := request.RequestInfoFrom(ctx)
			if longRunning = longRunningCheck(req, ri); longRunning {
				longRunningSink = sink
			}
		}
		respWriter := decorateResponseWriter(w, ev, longRunningSink)

		// send audit event when we leave this func, either via a panic or cleanly. In the case of long
		// running requests, this will be the second audit event.
		defer func() {
			if r := recover(); r != nil {
				ev.ResponseStatus = &metav1.Status{
					Code: http.StatusInternalServerError,
				}
				sink.ProcessEvents(ev)
				panic(r)
			}

			if ev.ResponseStatus == nil {
				ev.ResponseStatus = &metav1.Status{
					Code: 200,
				}
			}

			sink.ProcessEvents(ev)
		}()
		handler.ServeHTTP(respWriter, req)
	})
}

func decorateResponseWriter(responseWriter http.ResponseWriter, ev *auditinternal.Event, sink audit.Sink) http.ResponseWriter {
	delegate := &auditResponseWriter{
		ResponseWriter: responseWriter,
		event:          ev,
		sink:           sink,
	}

	// check if the ResponseWriter we're wrapping is the fancy one we need
	// or if the basic is sufficient
	_, cn := responseWriter.(http.CloseNotifier)
	_, fl := responseWriter.(http.Flusher)
	_, hj := responseWriter.(http.Hijacker)
	if cn && fl && hj {
		return &fancyResponseWriterDelegator{delegate}
	}
	return delegate
}

var _ http.ResponseWriter = &auditResponseWriter{}

// auditResponseWriter intercepts WriteHeader, sets it in the event. If the sink is set, it will
// create immediately an event (for long running requests).
type auditResponseWriter struct {
	http.ResponseWriter
	event *auditinternal.Event
	once  sync.Once
	sink  audit.Sink
}

func (a *auditResponseWriter) processCode(code int) {
	a.once.Do(func() {
		if a.sink != nil {
			a.sink.ProcessEvents(a.event)
		}

		// for now we use the ResponseStatus as marker that it's the first or second event
		// of a long running request. As soon as we have such a field in the event, we can
		// change this.
		if a.event.ResponseStatus == nil {
			a.event.ResponseStatus = &metav1.Status{}
		}
		a.event.ResponseStatus.Code = int32(code)
	})
}

func (a *auditResponseWriter) Write(bs []byte) (int, error) {
	a.processCode(200) // the Go library calls WriteHeader internally if no code was written yet. But this will go unnoticed for us
	return a.ResponseWriter.Write(bs)
}

func (a *auditResponseWriter) WriteHeader(code int) {
	a.processCode(code)
	a.ResponseWriter.WriteHeader(code)
}

// fancyResponseWriterDelegator implements http.CloseNotifier, http.Flusher and
// http.Hijacker which are needed to make certain http operation (e.g. watch, rsh, etc)
// working.
type fancyResponseWriterDelegator struct {
	*auditResponseWriter
}

func (f *fancyResponseWriterDelegator) CloseNotify() <-chan bool {
	return f.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

func (f *fancyResponseWriterDelegator) Flush() {
	f.ResponseWriter.(http.Flusher).Flush()
}

func (f *fancyResponseWriterDelegator) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return f.ResponseWriter.(http.Hijacker).Hijack()
}

var _ http.CloseNotifier = &fancyResponseWriterDelegator{}
var _ http.Flusher = &fancyResponseWriterDelegator{}
var _ http.Hijacker = &fancyResponseWriterDelegator{}
