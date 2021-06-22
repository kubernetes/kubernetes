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

package httplog

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"net/http"
	"runtime"
	"strings"
	"time"

	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

// StacktracePred returns true if a stacktrace should be logged for this status.
type StacktracePred func(httpStatus int) (logStacktrace bool)

type logger interface {
	Addf(format string, data ...interface{})
}

type respLoggerContextKeyType int

// respLoggerContextKey is used to store the respLogger pointer in the request context.
const respLoggerContextKey respLoggerContextKeyType = iota

// Add a layer on top of ResponseWriter, so we can track latency and error
// message sources.
//
// TODO now that we're using go-restful, we shouldn't need to be wrapping
// the http.ResponseWriter. We can recover panics from go-restful, and
// the logging value is questionable.
type respLogger struct {
	hijacked       bool
	statusRecorded bool
	status         int
	statusStack    string
	addedInfo      strings.Builder
	startTime      time.Time

	captureErrorOutput bool

	req *http.Request
	w   http.ResponseWriter

	logStacktracePred StacktracePred
}

// Simple logger that logs immediately when Addf is called
type passthroughLogger struct{}

//lint:ignore SA1019 Interface implementation check to make sure we don't drop CloseNotifier again
var _ http.CloseNotifier = &respLogger{}

// Addf logs info immediately.
func (passthroughLogger) Addf(format string, data ...interface{}) {
	klog.V(2).Info(fmt.Sprintf(format, data...))
}

// DefaultStacktracePred is the default implementation of StacktracePred.
func DefaultStacktracePred(status int) bool {
	return (status < http.StatusOK || status >= http.StatusInternalServerError) && status != http.StatusSwitchingProtocols
}

// WithLogging wraps the handler with logging.
func WithLogging(handler http.Handler, pred StacktracePred) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		if old := respLoggerFromRequest(req); old != nil {
			panic("multiple WithLogging calls!")
		}

		startTime := time.Now()
		if receivedTimestamp, ok := request.ReceivedTimestampFrom(ctx); ok {
			startTime = receivedTimestamp
		}

		rl := newLoggedWithStartTime(req, w, startTime).StacktraceWhen(pred)
		req = req.WithContext(context.WithValue(ctx, respLoggerContextKey, rl))

		if klog.V(3).Enabled() {
			defer rl.Log()
		}
		handler.ServeHTTP(rl, req)
	})
}

// respLoggerFromContext returns the respLogger or nil.
func respLoggerFromContext(ctx context.Context) *respLogger {
	val := ctx.Value(respLoggerContextKey)
	if rl, ok := val.(*respLogger); ok {
		return rl
	}
	return nil
}

func respLoggerFromRequest(req *http.Request) *respLogger {
	return respLoggerFromContext(req.Context())
}

func newLoggedWithStartTime(req *http.Request, w http.ResponseWriter, startTime time.Time) *respLogger {
	return &respLogger{
		startTime:         startTime,
		req:               req,
		w:                 w,
		logStacktracePred: DefaultStacktracePred,
	}
}

// newLogged turns a normal response writer into a logged response writer.
func newLogged(req *http.Request, w http.ResponseWriter) *respLogger {
	return newLoggedWithStartTime(req, w, time.Now())
}

// LogOf returns the logger hiding in w. If there is not an existing logger
// then a passthroughLogger will be created which will log to stdout immediately
// when Addf is called.
func LogOf(req *http.Request, w http.ResponseWriter) logger {
	if rl := respLoggerFromRequest(req); rl != nil {
		return rl
	}
	return &passthroughLogger{}
}

// Unlogged returns the original ResponseWriter, or w if it is not our inserted logger.
func Unlogged(req *http.Request, w http.ResponseWriter) http.ResponseWriter {
	if rl := respLoggerFromRequest(req); rl != nil {
		return rl.w
	}
	return w
}

// StacktraceWhen sets the stacktrace logging predicate, which decides when to log a stacktrace.
// There's a default, so you don't need to call this unless you don't like the default.
func (rl *respLogger) StacktraceWhen(pred StacktracePred) *respLogger {
	rl.logStacktracePred = pred
	return rl
}

// StatusIsNot returns a StacktracePred which will cause stacktraces to be logged
// for any status *not* in the given list.
func StatusIsNot(statuses ...int) StacktracePred {
	statusesNoTrace := map[int]bool{}
	for _, s := range statuses {
		statusesNoTrace[s] = true
	}
	return func(status int) bool {
		_, ok := statusesNoTrace[status]
		return !ok
	}
}

// Addf adds additional data to be logged with this request.
func (rl *respLogger) Addf(format string, data ...interface{}) {
	rl.addedInfo.WriteString("\n")
	rl.addedInfo.WriteString(fmt.Sprintf(format, data...))
}

func AddInfof(ctx context.Context, format string, data ...interface{}) {
	if rl := respLoggerFromContext(ctx); rl != nil {
		rl.Addf(format, data...)
	}
}

// Log is intended to be called once at the end of your request handler, via defer
func (rl *respLogger) Log() {
	latency := time.Since(rl.startTime)
	auditID := request.GetAuditIDTruncated(rl.req)

	verb := rl.req.Method
	if requestInfo, ok := request.RequestInfoFrom(rl.req.Context()); ok {
		// If we can find a requestInfo, we can get a scope, and then
		// we can convert GETs to LISTs when needed.
		scope := metrics.CleanScope(requestInfo)
		verb = metrics.CanonicalVerb(strings.ToUpper(verb), scope)
	}
	// mark APPLY requests and WATCH requests correctly.
	verb = metrics.CleanVerb(verb, rl.req)

	keysAndValues := []interface{}{
		"verb", verb,
		"URI", rl.req.RequestURI,
		"latency", latency,
		"userAgent", rl.req.UserAgent(),
		"audit-ID", auditID,
		"srcIP", rl.req.RemoteAddr,
	}

	if rl.hijacked {
		keysAndValues = append(keysAndValues, "hijacked", true)
	} else {
		keysAndValues = append(keysAndValues, "resp", rl.status)
		if len(rl.statusStack) > 0 {
			keysAndValues = append(keysAndValues, "statusStack", rl.statusStack)
		}
		info := rl.addedInfo.String()
		if len(info) > 0 {
			keysAndValues = append(keysAndValues, "addedInfo", info)
		}
	}

	klog.InfoSDepth(1, "HTTP", keysAndValues...)
}

// Header implements http.ResponseWriter.
func (rl *respLogger) Header() http.Header {
	return rl.w.Header()
}

// Write implements http.ResponseWriter.
func (rl *respLogger) Write(b []byte) (int, error) {
	if !rl.statusRecorded {
		rl.recordStatus(http.StatusOK) // Default if WriteHeader hasn't been called
	}
	if rl.captureErrorOutput {
		rl.Addf("logging error output: %q\n", string(b))
	}
	return rl.w.Write(b)
}

// Flush implements http.Flusher even if the underlying http.Writer doesn't implement it.
// Flush is used for streaming purposes and allows to flush buffered data to the client.
func (rl *respLogger) Flush() {
	if flusher, ok := rl.w.(http.Flusher); ok {
		flusher.Flush()
	} else if klog.V(2).Enabled() {
		klog.InfoDepth(1, fmt.Sprintf("Unable to convert %+v into http.Flusher", rl.w))
	}
}

// WriteHeader implements http.ResponseWriter.
func (rl *respLogger) WriteHeader(status int) {
	rl.recordStatus(status)
	rl.w.WriteHeader(status)
}

// Hijack implements http.Hijacker.
func (rl *respLogger) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	rl.hijacked = true
	return rl.w.(http.Hijacker).Hijack()
}

// CloseNotify implements http.CloseNotifier
func (rl *respLogger) CloseNotify() <-chan bool {
	//lint:ignore SA1019 There are places in the code base requiring the CloseNotifier interface to be implemented.
	return rl.w.(http.CloseNotifier).CloseNotify()
}

func (rl *respLogger) recordStatus(status int) {
	rl.status = status
	rl.statusRecorded = true
	if rl.logStacktracePred(status) {
		// Only log stacks for errors
		stack := make([]byte, 50*1024)
		stack = stack[:runtime.Stack(stack, false)]
		rl.statusStack = "\n" + string(stack)
		rl.captureErrorOutput = true
	} else {
		rl.statusStack = ""
	}
}
