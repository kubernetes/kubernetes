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
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
)

// WithAudit decorates a http.Handler with audit logging information for all the
// requests coming to the server. Audit level is decided according to requests'
// attributes and audit policy. Logs are emitted to the audit sink to
// process events. If sink or audit policy is nil, no decoration takes place.
func WithAudit(handler http.Handler, sink audit.Sink, policy audit.PolicyRuleEvaluator, longRunningCheck request.LongRunningRequestCheck) http.Handler {
	if sink == nil || policy == nil {
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		auditContext, err := evaluatePolicyAndCreateAuditEvent(req, policy)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to create audit event: %v", err))
			responsewriters.InternalError(w, req, errors.New("failed to create audit event"))
			return
		}

		ev := auditContext.Event
		if ev == nil || req.Context() == nil {
			handler.ServeHTTP(w, req)
			return
		}

		req = req.WithContext(audit.WithAuditContext(req.Context(), auditContext))

		ctx := req.Context()
		omitStages := auditContext.RequestAuditConfig.OmitStages

		ev.Stage = auditinternal.StageRequestReceived
		if processed := processAuditEvent(ctx, sink, ev, omitStages); !processed {
			audit.ApiserverAuditDroppedCounter.WithContext(ctx).Inc()
			responsewriters.InternalError(w, req, errors.New("failed to store audit event"))
			return
		}

		// intercept the status code
		var longRunningSink audit.Sink
		if longRunningCheck != nil {
			ri, _ := request.RequestInfoFrom(ctx)
			if longRunningCheck(req, ri) {
				longRunningSink = sink
			}
		}
		respWriter := decorateResponseWriter(ctx, w, ev, longRunningSink, omitStages)

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
				processAuditEvent(ctx, sink, ev, omitStages)
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
				processAuditEvent(ctx, longRunningSink, ev, omitStages)
			}

			ev.Stage = auditinternal.StageResponseComplete
			if ev.ResponseStatus == nil {
				ev.ResponseStatus = fakedSuccessStatus
			}
			processAuditEvent(ctx, sink, ev, omitStages)
		}()
		handler.ServeHTTP(respWriter, req)
	})
}

// evaluatePolicyAndCreateAuditEvent is responsible for evaluating the audit
// policy configuration applicable to the request and create a new audit
// event that will be written to the API audit log.
// - error if anything bad happened
func evaluatePolicyAndCreateAuditEvent(req *http.Request, policy audit.PolicyRuleEvaluator) (*audit.AuditContext, error) {
	ctx := req.Context()

	attribs, err := GetAuthorizerAttributes(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to GetAuthorizerAttributes: %v", err)
	}

	ls := policy.EvaluatePolicyRule(attribs)
	audit.ObservePolicyLevel(ctx, ls.Level)
	if ls.Level == auditinternal.LevelNone {
		// Don't audit.
		return &audit.AuditContext{
			RequestAuditConfig: ls.RequestAuditConfig,
		}, nil
	}

	requestReceivedTimestamp, ok := request.ReceivedTimestampFrom(ctx)
	if !ok {
		requestReceivedTimestamp = time.Now()
	}
	ev, err := audit.NewEventFromRequest(req, requestReceivedTimestamp, ls.Level, attribs)
	if err != nil {
		return nil, fmt.Errorf("failed to complete audit event from request: %v", err)
	}

	return &audit.AuditContext{
		RequestAuditConfig: ls.RequestAuditConfig,
		Event:              ev,
	}, nil
}

func processAuditEvent(ctx context.Context, sink audit.Sink, ev *auditinternal.Event, omitStages []auditinternal.Stage) bool {
	for _, stage := range omitStages {
		if ev.Stage == stage {
			return true
		}
	}

	if ev.Stage == auditinternal.StageRequestReceived {
		ev.StageTimestamp = metav1.NewMicroTime(ev.RequestReceivedTimestamp.Time)
	} else {
		ev.StageTimestamp = metav1.NewMicroTime(time.Now())
	}
	audit.ObserveEvent(ctx)
	return sink.ProcessEvents(ev)
}

func decorateResponseWriter(ctx context.Context, responseWriter http.ResponseWriter, ev *auditinternal.Event, sink audit.Sink, omitStages []auditinternal.Stage) http.ResponseWriter {
	delegate := &auditResponseWriter{
		ctx:            ctx,
		ResponseWriter: responseWriter,
		event:          ev,
		sink:           sink,
		omitStages:     omitStages,
	}

	return responsewriter.WrapForHTTP1Or2(delegate)
}

var _ http.ResponseWriter = &auditResponseWriter{}
var _ responsewriter.UserProvidedDecorator = &auditResponseWriter{}

// auditResponseWriter intercepts WriteHeader, sets it in the event. If the sink is set, it will
// create immediately an event (for long running requests).
type auditResponseWriter struct {
	http.ResponseWriter
	ctx        context.Context
	event      *auditinternal.Event
	once       sync.Once
	sink       audit.Sink
	omitStages []auditinternal.Stage
}

func (a *auditResponseWriter) Unwrap() http.ResponseWriter {
	return a.ResponseWriter
}

func (a *auditResponseWriter) processCode(code int) {
	a.once.Do(func() {
		if a.event.ResponseStatus == nil {
			a.event.ResponseStatus = &metav1.Status{}
		}
		a.event.ResponseStatus.Code = int32(code)
		a.event.Stage = auditinternal.StageResponseStarted

		if a.sink != nil {
			processAuditEvent(a.ctx, a.sink, a.event, a.omitStages)
		}
	})
}

func (a *auditResponseWriter) Write(bs []byte) (int, error) {
	// the Go library calls WriteHeader internally if no code was written yet. But this will go unnoticed for us
	a.processCode(http.StatusOK)
	return a.ResponseWriter.Write(bs)
}

func (a *auditResponseWriter) WriteHeader(code int) {
	a.processCode(code)
	a.ResponseWriter.WriteHeader(code)
}

func (a *auditResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	// fake a response status before protocol switch happens
	a.processCode(http.StatusSwitchingProtocols)

	// the outer ResponseWriter object returned by WrapForHTTP1Or2 implements
	// http.Hijacker if the inner object (a.ResponseWriter) implements http.Hijacker.
	return a.ResponseWriter.(http.Hijacker).Hijack()
}
