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
	"fmt"
	"net"
	"net/http"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// WithAudit decorates a http.Handler with audit logging information for all the
// requests coming to the server. Audit level is decided according to requests'
// attributes and audit policy. Logs are emitted to the audit sink to
// process events. If sink or audit policy is nil, no decoration takes place.
func WithAudit(handler http.Handler, requestContextMapper request.RequestContextMapper, sink audit.Sink, policy policy.Checker, longRunningCheck request.LongRunningRequestCheck) http.Handler {
	if sink == nil || policy == nil {
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

		level := policy.Level(attribs)
		audit.ObservePolicyLevel(level)
		if level == auditinternal.LevelNone {
			// Don't audit.
			handler.ServeHTTP(w, req)
			return
		}

		ev, err := audit.NewEventFromRequest(req, level, attribs)
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

		ev.Stage = auditinternal.StageRequestReceived
		processEvent(sink, ev)

		// intercept the status code
		var longRunningSink audit.Sink
		if longRunningCheck != nil {
			ri, _ := request.RequestInfoFrom(ctx)
			if longRunningCheck(req, ri) {
				longRunningSink = sink
			}
		}
		respWriter := decorateResponseWriter(w, ev, longRunningSink)

		// send audit event when we leave this func, either via a panic or cleanly. In the case of long
		// running requests, this will be the second audit event.
		defer func() {
			if r := recover(); r != nil {
				defer panic(r)
				ev.Stage = auditinternal.StagePanic
				ev.ResponseStatus = &metav1.Status{
					Code:    http.StatusInternalServerError,
					Status:  metav1.StatusFailure,
					Reason:  metav1.StatusReasonInternalError,
					Message: fmt.Sprintf("APIServer panic'd: %v", r),
				}
				processEvent(sink, ev)
				return
			}

			// if no StageResponseStarted event was sent b/c neither a status code nor a body was sent, fake it here
			// But Audit-Id http header will only be sent when http.ResponseWriter.WriteHeader is called.
			fakedSuccessStatus := &metav1.Status{
				Code:    http.StatusOK,
				Status:  metav1.StatusSuccess,
				Message: "Connection closed early",
			}
			if ev.ResponseStatus == nil && longRunningSink != nil {
				ev.ResponseStatus = fakedSuccessStatus
				ev.Stage = auditinternal.StageResponseStarted
				processEvent(longRunningSink, ev)
			}

			ev.Stage = auditinternal.StageResponseComplete
			if ev.ResponseStatus == nil {
				ev.ResponseStatus = fakedSuccessStatus
			}
			processEvent(sink, ev)
		}()
		handler.ServeHTTP(respWriter, req)
	})
}

func processEvent(sink audit.Sink, ev *auditinternal.Event) {
	audit.ObserveEvent()
	sink.ProcessEvents(ev)
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

func (a *auditResponseWriter) setHttpHeader() {
	a.ResponseWriter.Header().Set(auditinternal.HeaderAuditID, string(a.event.AuditID))
}

func (a *auditResponseWriter) processCode(code int) {
	a.once.Do(func() {
		if a.event.ResponseStatus == nil {
			a.event.ResponseStatus = &metav1.Status{}
		}
		a.event.ResponseStatus.Code = int32(code)
		a.event.Stage = auditinternal.StageResponseStarted

		if a.sink != nil {
			processEvent(a.sink, a.event)
		}
	})
}

func (a *auditResponseWriter) Write(bs []byte) (int, error) {
	// the Go library calls WriteHeader internally if no code was written yet. But this will go unnoticed for us
	a.processCode(http.StatusOK)
	a.setHttpHeader()

	return a.ResponseWriter.Write(bs)
}

func (a *auditResponseWriter) WriteHeader(code int) {
	a.processCode(code)
	a.setHttpHeader()
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
	// fake a response status before protocol switch happens
	f.processCode(http.StatusSwitchingProtocols)

	// This will be ignored if WriteHeader() function has aready been called.
	// It's not guaranteed Audit-ID http header is sent for all requests.
	// For example, when user run "kubectl exec", apiserver uses a proxy handler
	// to deal with the request, users can only get http headers returned by kubelet node.
	f.setHttpHeader()

	return f.ResponseWriter.(http.Hijacker).Hijack()
}

var _ http.CloseNotifier = &fancyResponseWriterDelegator{}
var _ http.Flusher = &fancyResponseWriterDelegator{}
var _ http.Hijacker = &fancyResponseWriterDelegator{}
