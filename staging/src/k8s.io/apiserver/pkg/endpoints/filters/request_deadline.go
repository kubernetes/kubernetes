/*
Copyright 2020 The Kubernetes Authors.

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
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

const (
	// The 'timeout' query parameter in the request URL has an invalid duration specified
	invalidTimeoutInURL = "invalid timeout specified in the request URL"
)

// WithRequestDeadline determines the timeout duration applicable to the given
// request and sets a new context with the appropriate deadline.
//
// auditWrapper: provides an http.Handler that audits a failed request.
// longRunning: returns true if he given request is a long running request.
// requestTimeout: the value obtained from the server run option
// 'request-timeout', if specified, all requests except those which match the
// longRunning predicate will timeout after this duration.
// minRequestTimeout: the value obtained from the server run option
// '--min-request-timeout', if specified (in seconds), long running requests
// such as watch will be allocated a random timeout between this value,
// and twice this value.
func WithRequestDeadline(handler http.Handler, sink audit.Sink, policy audit.PolicyRuleEvaluator, longRunning request.LongRunningRequestCheck,
	negotiatedSerializer runtime.NegotiatedSerializer, requestTimeout time.Duration, minRequestTimeout int) http.Handler {
	parser := &timeoutParser{
		watchReqDefaultTimeout: time.Duration(minRequestTimeout) * time.Second,
		shortReqDefaultTimeout: requestTimeout,
	}
	return withRequestDeadline(handler, sink, policy, longRunning, negotiatedSerializer, clock.RealClock{}, parser)
}

type timeoutParser struct {
	// If specified, this is the default timeout for all requests except
	// those which match the LongRunningRequestCheck predicate.
	shortReqDefaultTimeout time.Duration

	// If specified, by default the WATCH requests will be allocated
	// a random timeout between this value, and twice this value.
	watchReqDefaultTimeout time.Duration
}

// Parse determines the effective timeout duration for the given request.
// NOTE: long running requests (which match the LongRunningRequestCheck
// predicate) excluding WATCH are outside the scope.
//
// WATCH timeout:
// a) use the value in 'timeoutSeconds' (in seconds), if specified by the user.
// b) if the value from a is zero, then apply the default:
// timeout = seconds(min-request-timeout) *  ( 1 + random([0.0,1.0)) )
//
// Short request timeout:
// a) use the duration in 'timeout' request parameter, if specified by the user.
// b) clamp, if the duration from a exceeds the maximum allowed timeout = max(timeout, --request-timeout)
// c) if the value from a is zero, apply default: timeout = --request-timeout
func (tp timeoutParser) Parse(req *http.Request, reqInfo *request.RequestInfo) (time.Duration, bool, error) {
	if reqInfo.IsWatch() {
		timeoutSeconds := parseWatchTimeout(req)
		timeout := request.GetTimeoutForWatch(timeoutSeconds, tp.watchReqDefaultTimeout)
		return timeout, true, nil
	}

	userSpecifiedTimeout, ok, err := parseTimeout(req)
	if err != nil {
		return 0, false, err
	}
	// we use the default timeout enforced by the apiserver:
	// - if the user has specified a timeout of 0s, this implies no timeout on the user's part.
	// - if the user has specified a timeout that exceeds the maximum deadline allowed by the apiserver.
	timeout := tp.shortReqDefaultTimeout
	if ok && userSpecifiedTimeout > 0 && userSpecifiedTimeout < tp.shortReqDefaultTimeout {
		timeout = userSpecifiedTimeout
	}
	return timeout, true, nil
}

func parseWatchTimeout(req *http.Request) *int64 {
	// we don't return error due to Decodeparameters failing to decode the
	// url parameters, or an invalid value specified in 'timeoutSeconds',
	// this is in keeping with the current (1.30) behavior.
	opts := metainternalversion.ListOptions{}
	if err := metainternalversionscheme.ParameterCodec.DecodeParameters(req.URL.Query(), metav1.SchemeGroupVersion, &opts); err != nil {
		opts = metainternalversion.ListOptions{}
		// TODO: Currently we explicitly ignore ?timeout= and use only ?timeoutSeconds=.
		if values := req.URL.Query()["timeoutSeconds"]; len(values) > 0 {
			if ts, err := strconv.Atoi(values[0]); err == nil {
				opts.TimeoutSeconds = ptr.To(int64(ts))
			}
		}
	}
	return opts.TimeoutSeconds
}

func withRequestDeadline(handler http.Handler, sink audit.Sink, policy audit.PolicyRuleEvaluator, longRunning request.LongRunningRequestCheck,
	negotiatedSerializer runtime.NegotiatedSerializer, clock clock.PassiveClock, parser *timeoutParser) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()

		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			handleError(w, req, http.StatusInternalServerError, fmt.Errorf("no RequestInfo found in context, handler chain must be wrong"))
			return
		}

		// long running requests (except for WATCH requests) are
		// outside the scope of deadline bound context.
		if longRunning(req, requestInfo) && !requestInfo.IsWatch() {
			handler.ServeHTTP(w, req)
			return
		}

		timeout, ok, err := parser.Parse(req, requestInfo)
		if err != nil {
			statusErr := apierrors.NewBadRequest(err.Error())

			klog.Errorf("Error - %s: %#v", err.Error(), req.RequestURI)

			failed := failedErrorHandler(negotiatedSerializer, statusErr)
			failWithAudit := withFailedRequestAudit(failed, statusErr, sink, policy)
			failWithAudit.ServeHTTP(w, req)
			return
		}
		if !ok {
			handler.ServeHTTP(w, req)
			return
		}

		started := clock.Now()
		if requestStartedTimestamp, ok := request.ReceivedTimestampFrom(ctx); ok {
			started = requestStartedTimestamp
		}

		ctx, cancel := context.WithDeadline(ctx, started.Add(timeout))
		defer cancel()

		req = req.WithContext(ctx)
		handler.ServeHTTP(w, req)
	})
}

// withFailedRequestAudit decorates a failed http.Handler and is used to audit a failed request.
// statusErr is used to populate the Message property of ResponseStatus.
func withFailedRequestAudit(failedHandler http.Handler, statusErr *apierrors.StatusError, sink audit.Sink, policy audit.PolicyRuleEvaluator) http.Handler {
	if sink == nil || policy == nil {
		return failedHandler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ac, err := evaluatePolicyAndCreateAuditEvent(req, policy)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to create audit event: %v", err))
			responsewriters.InternalError(w, req, errors.New("failed to create audit event"))
			return
		}

		if !ac.Enabled() {
			failedHandler.ServeHTTP(w, req)
			return
		}
		ev := &ac.Event

		ev.ResponseStatus = &metav1.Status{}
		ev.Stage = auditinternal.StageResponseStarted
		if statusErr != nil {
			ev.ResponseStatus.Message = statusErr.Error()
		}

		rw := decorateResponseWriter(req.Context(), w, ev, sink, ac.RequestAuditConfig.OmitStages)
		failedHandler.ServeHTTP(rw, req)
	})
}

// failedErrorHandler returns an http.Handler that uses the specified StatusError object
// to render an error response to the request.
func failedErrorHandler(s runtime.NegotiatedSerializer, statusError *apierrors.StatusError) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		requestInfo, found := request.RequestInfoFrom(ctx)
		if !found {
			responsewriters.InternalError(w, req, errors.New("no RequestInfo found in the context"))
			return
		}

		gv := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		responsewriters.ErrorNegotiated(statusError, s, gv, w, req)
	})
}

// parseTimeout parses the given HTTP request URL and extracts the timeout query parameter
// value if specified by the user.
// If a timeout is not specified the function returns false and err is set to nil
// If the value specified is malformed then the function returns false and err is set
func parseTimeout(req *http.Request) (time.Duration, bool, error) {
	value := req.URL.Query().Get("timeout")
	if value == "" {
		return 0, false, nil
	}

	timeout, err := time.ParseDuration(value)
	if err != nil {
		return 0, false, fmt.Errorf("%s - %s", invalidTimeoutInURL, err.Error())
	}

	return timeout, true, nil
}

func handleError(w http.ResponseWriter, r *http.Request, code int, err error) {
	errorMsg := fmt.Sprintf("Error - %s: %#v", err.Error(), r.RequestURI)
	http.Error(w, errorMsg, code)
	klog.Errorf(errorMsg)
}
