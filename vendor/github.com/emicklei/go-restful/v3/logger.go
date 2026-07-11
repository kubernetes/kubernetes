package restful

// Copyright 2014 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.
import (
	"github.com/emicklei/go-restful/v3/log"
)

var trace bool = false
var traceLogger log.StdLogger

func init() {
	traceLogger = log.Logger // use the package logger by default
}

// TraceLogger enables detailed logging of Http request matching and filter invocation. Default no logger is set.
// You may call EnableTracing() directly to enable trace logging to the package-wide logger.
func TraceLogger(logger log.StdLogger) {
	traceLogger = logger
	EnableTracing(logger != nil)
}

// SetLogger exposes the setter for the global logger on the top-level package
func SetLogger(customLogger log.StdLogger) {
	log.SetLogger(customLogger)
}

// EnableTracing can be used to Trace logging on and off.
func EnableTracing(enabled bool) {
	trace = enabled
}
