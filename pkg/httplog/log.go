/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"net"
	"net/http"
	"runtime"
	"time"

	"github.com/golang/glog"
)

// Handler wraps all HTTP calls to delegate with nice logging.
// delegate may use LogOf(w).Addf(...) to write additional info to
// the per-request log message.
//
// Intended to wrap calls to your ServeMux.
func Handler(delegate http.Handler, pred StacktracePred) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer NewLogged(req, &w).StacktraceWhen(pred).Log()
		delegate.ServeHTTP(w, req)
	})
}

// StacktracePred returns true if a stacktrace should be logged for this status.
type StacktracePred func(httpStatus int) (logStacktrace bool)

type logger interface {
	Addf(format string, data ...interface{})
}

// Add a layer on top of ResponseWriter, so we can track latency and error
// message sources.
//
// TODO now that we're using go-restful, we shouldn't need to be wrapping
// the http.ResponseWriter. We can recover panics from go-restful, and
// the logging value is questionable.
type respLogger struct {
	status      int
	statusStack string
	addedInfo   string
	startTime   time.Time

	req *http.Request
	w   http.ResponseWriter

	logStacktracePred StacktracePred
}

// Simple logger that logs immediately when Addf is called
type passthroughLogger struct{}

// Addf logs info immediately.
func (passthroughLogger) Addf(format string, data ...interface{}) {
	glog.InfoDepth(1, fmt.Sprintf(format, data...))
}

// DefaultStacktracePred is the default implementation of StacktracePred.
func DefaultStacktracePred(status int) bool {
	return (status < http.StatusOK || status >= http.StatusInternalServerError) && status != http.StatusSwitchingProtocols
}

// NewLogged turns a normal response writer into a logged response writer.
//
// Usage:
//
// defer NewLogged(req, &w).StacktraceWhen(StatusIsNot(200, 202)).Log()
//
// (Only the call to Log() is deferred, so you can set everything up in one line!)
//
// Note that this *changes* your writer, to route response writing actions
// through the logger.
//
// Use LogOf(w).Addf(...) to log something along with the response result.
func NewLogged(req *http.Request, w *http.ResponseWriter) *respLogger {
	if _, ok := (*w).(*respLogger); ok {
		// Don't double-wrap!
		panic("multiple NewLogged calls!")
	}
	rl := &respLogger{
		startTime:         time.Now(),
		req:               req,
		w:                 *w,
		logStacktracePred: DefaultStacktracePred,
	}
	*w = rl // hijack caller's writer!
	return rl
}

// LogOf returns the logger hiding in w. If there is not an existing logger
// then a passthroughLogger will be created which will log to stdout immediately
// when Addf is called.
func LogOf(req *http.Request, w http.ResponseWriter) logger {
	if _, exists := w.(*respLogger); !exists {
		pl := &passthroughLogger{}
		return pl
	}
	if rl, ok := w.(*respLogger); ok {
		return rl
	}
	panic("Unable to find or create the logger!")
}

// Unlogged returns the original ResponseWriter, or w if it is not our inserted logger.
func Unlogged(w http.ResponseWriter) http.ResponseWriter {
	if rl, ok := w.(*respLogger); ok {
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
	return func(status int) bool {
		for _, s := range statuses {
			if status == s {
				return false
			}
		}
		return true
	}
}

// Addf adds additional data to be logged with this request.
func (rl *respLogger) Addf(format string, data ...interface{}) {
	rl.addedInfo += "\n" + fmt.Sprintf(format, data...)
}

// Log is intended to be called once at the end of your request handler, via defer
func (rl *respLogger) Log() {
	latency := time.Since(rl.startTime)
	if glog.V(2) {
		glog.InfoDepth(1, fmt.Sprintf("%s %s: (%v) %v%v%v [%s %s]", rl.req.Method, rl.req.RequestURI, latency, rl.status, rl.statusStack, rl.addedInfo, rl.req.Header["User-Agent"], rl.req.RemoteAddr))
	}
}

// Header implements http.ResponseWriter.
func (rl *respLogger) Header() http.Header {
	return rl.w.Header()
}

// Write implements http.ResponseWriter.
func (rl *respLogger) Write(b []byte) (int, error) {
	return rl.w.Write(b)
}

// Flush implements http.Flusher even if the underlying http.Writer doesn't implement it.
// Flush is used for streaming purposes and allows to flush buffered data to the client.
func (rl *respLogger) Flush() {
	if flusher, ok := rl.w.(http.Flusher); ok {
		flusher.Flush()
	} else if glog.V(2) {
		glog.InfoDepth(1, fmt.Sprintf("Unable to convert %+v into http.Flusher", rl.w))
	}
}

// WriteHeader implements http.ResponseWriter.
func (rl *respLogger) WriteHeader(status int) {
	rl.status = status
	if rl.logStacktracePred(status) {
		// Only log stacks for errors
		stack := make([]byte, 2048)
		stack = stack[:runtime.Stack(stack, false)]
		rl.statusStack = "\n" + string(stack)
	} else {
		rl.statusStack = ""
	}
	rl.w.WriteHeader(status)
}

// Hijack implements http.Hijacker.
func (rl *respLogger) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return rl.w.(http.Hijacker).Hijack()
}
