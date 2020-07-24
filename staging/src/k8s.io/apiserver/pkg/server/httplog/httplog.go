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
	"time"

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
	addedInfo      string
	startTime      time.Time
	isTerminating  bool

	captureErrorOutput bool

	req *http.Request
	w   http.ResponseWriter

	logStacktracePred StacktracePred
}

// Simple logger that logs immediately when Addf is called
type passthroughLogger struct{}

// Addf logs info immediately.
func (passthroughLogger) Addf(format string, data ...interface{}) {
	klog.V(2).Info(fmt.Sprintf(format, data...))
}

// DefaultStacktracePred is the default implementation of StacktracePred.
func DefaultStacktracePred(status int) bool {
	return (status < http.StatusOK || status >= http.StatusInternalServerError) && status != http.StatusSwitchingProtocols
}

// WithLogging wraps the handler with logging.
func WithLogging(handler http.Handler, pred StacktracePred, isTerminatingFn func() bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		if old := respLoggerFromContext(req); old != nil {
			panic("multiple WithLogging calls!")
		}
		isTerminating := false
		if isTerminatingFn != nil {
			isTerminating = isTerminatingFn()
		}
		rl := newLogged(req, w).StacktraceWhen(pred).IsTerminating(isTerminating)
		req = req.WithContext(context.WithValue(ctx, respLoggerContextKey, rl))

		if klog.V(3).Enabled() || (rl.isTerminating && klog.V(1).Enabled()) {
			defer func() { klog.InfoS("HTTP", rl.LogArgs()...) }()
		}
		handler.ServeHTTP(rl, req)
	})
}

// respLoggerFromContext returns the respLogger or nil.
func respLoggerFromContext(req *http.Request) *respLogger {
	ctx := req.Context()
	val := ctx.Value(respLoggerContextKey)
	if rl, ok := val.(*respLogger); ok {
		return rl
	}
	return nil
}

// newLogged turns a normal response writer into a logged response writer.
func newLogged(req *http.Request, w http.ResponseWriter) *respLogger {
	return &respLogger{
		startTime:         time.Now(),
		req:               req,
		w:                 w,
		logStacktracePred: DefaultStacktracePred,
	}
}

// LogOf returns the logger hiding in w. If there is not an existing logger
// then a passthroughLogger will be created which will log to stdout immediately
// when Addf is called.
func LogOf(req *http.Request, w http.ResponseWriter) logger {
	if rl := respLoggerFromContext(req); rl != nil {
		return rl
	}
	return &passthroughLogger{}
}

// Unlogged returns the original ResponseWriter, or w if it is not our inserted logger.
func Unlogged(req *http.Request, w http.ResponseWriter) http.ResponseWriter {
	if rl := respLoggerFromContext(req); rl != nil {
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

// IsTerminating informs the logger that the server is terminating.
func (rl *respLogger) IsTerminating(is bool) *respLogger {
	rl.isTerminating = is
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
	rl.addedInfo += "\n" + fmt.Sprintf(format, data...)
}

func (rl *respLogger) LogArgs() []interface{} {
	latency := time.Since(rl.startTime)
	if rl.hijacked {
		return []interface{}{
			"verb", rl.req.Method,
			"URI", rl.req.RequestURI,
			"latency", latency,
			"userAgent", rl.req.UserAgent(),
			"srcIP", rl.req.RemoteAddr,
			"hijacked", true,
		}
	}
	args := []interface{}{
		"verb", rl.req.Method,
		"URI", rl.req.RequestURI,
		"latency", latency,
		"userAgent", rl.req.UserAgent(),
		"srcIP", rl.req.RemoteAddr,
		"resp", rl.status,
	}
	if len(rl.statusStack) > 0 {
		args = append(args, "statusStack", rl.statusStack)
	}

	if len(rl.addedInfo) > 0 {
		args = append(args, "addedInfo", rl.addedInfo)
	}
	return args
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
