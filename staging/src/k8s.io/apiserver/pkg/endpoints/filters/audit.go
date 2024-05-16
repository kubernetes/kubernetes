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
		ac, err := evaluatePolicyAndCreateAuditEvent(req, policy)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to create audit event: %v", err))
			responsewriters.InternalError(w, req, errors.New("failed to create audit event"))
			return
		}

		if !ac.Enabled() {
			handler.ServeHTTP(w, req)
			return
		}
		ev := &ac.Event

		ctx := req.Context()
		omitStages := ac.RequestAuditConfig.OmitStages

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
	ac := audit.AuditContextFrom(ctx)
	if ac == nil {
		// Auditing not configured.
		return nil, nil
	}

	attribs, err := GetAuthorizerAttributes(ctx)
	if err != nil {
		return ac, fmt.Errorf("failed to GetAuthorizerAttributes: %v", err)
	}

	rac := policy.EvaluatePolicyRule(attribs)
	audit.ObservePolicyLevel(ctx, rac.Level)
	ac.RequestAuditConfig = rac
	if rac.Level == auditinternal.LevelNone {
		// Don't audit.
		return ac, nil
	}

	requestReceivedTimestamp, ok := request.ReceivedTimestampFrom(ctx)
	if !ok {
		requestReceivedTimestamp = time.Now()
	}
	audit.LogRequestMetadata(ctx, req, requestReceivedTimestamp, rac.Level, attribs)

	return ac, nil
}

// writeLatencyToAnnotation writes the latency incurred in different
// layers of the apiserver to the annotations of the audit object.
// it should be invoked after ev.StageTimestamp has been set appropriately.
func writeLatencyToAnnotation(ctx context.Context, ev *auditinternal.Event) {
	// we will track latency in annotation only when the total latency
	// of the given request exceeds 500ms, this is in keeping with the
	// traces in rest/handlers for create, delete, update,
	// get, list, and deletecollection.
	const threshold = 500 * time.Millisecond
	latency := ev.StageTimestamp.Time.Sub(ev.RequestReceivedTimestamp.Time)
	if latency <= threshold {
		return
	}

	// if we are tracking latency incurred inside different layers within
	// the apiserver, add these as annotation to the audit event object.
	layerLatencies := request.AuditAnnotationsFromLatencyTrackers(ctx)
	if len(layerLatencies) == 0 {
		// latency tracking is not enabled for this request
		return
	}

	// record the total latency for this request, for convenience.
	layerLatencies["apiserver.latency.k8s.io/total"] = latency.String()
	audit.AddAuditAnnotationsMap(ctx, layerLatencies)
}

func processAuditEvent(ctx context.Context, sink audit.Sink, ev *auditinternal.Event, omitStages []auditinternal.Stage) bool {
	for _, stage := range omitStages {
		if ev.Stage == stage {
			return true
		}
	}

	switch {
	case ev.Stage == auditinternal.StageRequestReceived:
		ev.StageTimestamp = metav1.NewMicroTime(ev.RequestReceivedTimestamp.Time)
	case ev.Stage == auditinternal.StageResponseComplete:
		ev.StageTimestamp = metav1.NewMicroTime(time.Now())
		writeLatencyToAnnotation(ctx, ev)
	default:
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
