/*
Copyright 2014 Google Inc. All rights reserved.

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
	"fmt"
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

// Add a layer on top of ResponseWriter, so we can track latency and error
// message sources.
type respLogger struct {
	status      int
	statusStack string
	addedInfo   string
	startTime   time.Time

	req *http.Request
	w   http.ResponseWriter

	logStacktracePred StacktracePred
}

// DefaultStacktracePred is the default implementation of StacktracePred.
func DefaultStacktracePred(status int) bool {
	return status < http.StatusOK || status >= http.StatusBadRequest
}

// NewLogged turns a normal response writer into a logged response writer.
//
// Usage:
//
// defer NewLogged(req, &w).StacktraceWhen(StatusIsNot(200, 202)).Log()
//
// (Only the call to Log() is defered, so you can set everything up in one line!)
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

// LogOf returns the logger hiding in w. Panics if there isn't such a logger,
// because NewLogged() must have been previously called for the log to work.
func LogOf(w http.ResponseWriter) *respLogger {
	if rl, ok := w.(*respLogger); ok {
		return rl
	}
	panic("Logger not installed yet!")
}

// Returns the existing logger hiding in w. If there is not an existing logger
// then one will be created.
func FindOrCreateLogOf(req *http.Request, w *http.ResponseWriter) *respLogger {
	if _, exists := (*w).(*respLogger); !exists {
		NewLogged(req, w)
	}
	return LogOf(*w)
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
	glog.Infof("%s %s: (%v) %v%v%v", rl.req.Method, rl.req.RequestURI, latency, rl.status, rl.statusStack, rl.addedInfo)
}

// Header implements http.ResponseWriter.
func (rl *respLogger) Header() http.Header {
	return rl.w.Header()
}

// Write implements http.ResponseWriter.
func (rl *respLogger) Write(b []byte) (int, error) {
	return rl.w.Write(b)
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
