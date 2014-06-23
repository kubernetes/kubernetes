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

package apiserver

import (
	"fmt"
	"log"
	"net/http"
	"runtime"
	"time"
)

// Add a layer on top of ResponseWriter, so we can track latency and error
// message sources.
type respLogger struct {
	status      int
	statusStack string
	addedInfo   string
	startTime   time.Time

	req *http.Request
	w   http.ResponseWriter
}

// Usage:
// logger := MakeLogged(req, w)
// w = logger // Route response writing actions through w
// defer logger.Log()
func MakeLogged(req *http.Request, w http.ResponseWriter) *respLogger {
	return &respLogger{
		startTime: time.Now(),
		req:       req,
		w:         w,
	}
}

// Add additional data to be logged with this request.
func (rl *respLogger) Addf(format string, data ...interface{}) {
	rl.addedInfo += "\n" + fmt.Sprintf(format, data...)
}

// Log is intended to be called once at the end of your request handler, via defer
func (rl *respLogger) Log() {
	latency := time.Since(rl.startTime)
	log.Printf("%s %s: (%v) %v%v%v", rl.req.Method, rl.req.RequestURI, latency, rl.status, rl.statusStack, rl.addedInfo)
}

// Implement http.ResponseWriter
func (rl *respLogger) Header() http.Header {
	return rl.w.Header()
}

// Implement http.ResponseWriter
func (rl *respLogger) Write(b []byte) (int, error) {
	return rl.w.Write(b)
}

// Implement http.ResponseWriter
func (rl *respLogger) WriteHeader(status int) {
	rl.status = status
	if status != http.StatusOK && status != http.StatusAccepted {
		// Only log stacks for errors
		stack := make([]byte, 2048)
		stack = stack[:runtime.Stack(stack, false)]
		rl.statusStack = "\n" + string(stack)
	} else {
		rl.statusStack = ""
	}
	rl.w.WriteHeader(status)
}
