/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package audit

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/pborman/uuid"

	"k8s.io/kubernetes/pkg/api"
	genericaudit "k8s.io/kubernetes/pkg/genericapiserver/audit"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

// auditResponseWriter implements http.ResponseWriter interface.
type auditResponseWriter struct {
	http.ResponseWriter
	out   io.Writer
	event *genericaudit.Event
}

func (a *auditResponseWriter) WriteHeader(code int) {
	a.event.Response = code
	a.ResponseWriter.WriteHeader(code)
}

var _ http.ResponseWriter = &auditResponseWriter{}

// fancyResponseWriterDelegator implements http.CloseNotifier, http.Flusher and
// http.Hijacker which are needed to make certain http operation (eg. watch, rsh, etc)
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
	return f.ResponseWriter.(http.Hijacker).Hijack()
}

var _ http.CloseNotifier = &fancyResponseWriterDelegator{}
var _ http.Flusher = &fancyResponseWriterDelegator{}
var _ http.Hijacker = &fancyResponseWriterDelegator{}

// WithAudit is responsible for logging audit information for all the
// request coming to server. Each audit log contains two entries:
// 1. the request line containing:
//    - unique id allowing to match the response line (see 2)
//    - source ip of the request
//    - HTTP method being invoked
//    - original user invoking the operation
//    - impersonated user for the operation
//    - namespace of the request or <none>
//    - uri is the full URI as requested
// 2. the response line containing:
//    - the unique id from 1
//    - response code
func WithAudit(handler http.Handler, requestContextMapper api.RequestContextMapper, out io.Writer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, _ := requestContextMapper.Get(req)

		ev := genericaudit.Event{
			Level:  genericaudit.StorageLogLevel,
			ID:     uuid.NewRandom().String(),
			Method: req.Method,
			URI:    req.URL.String(),
		}

		user, _ := api.UserFrom(ctx)
		ev.User = user.GetName()

		ev.AsUser = req.Header.Get("Impersonate-User")
		if len(ev.AsUser) == 0 {
			ev.AsUser = ev.User
		}

		ev.Namespace = api.NamespaceValue(ctx)
		if len(ev.Namespace) == 0 {
			ev.Namespace = "<none>"
		}

		requestContextMapper.Update(req, context.WithValue(ctx, genericaudit.EventContextKey, &ev))

		fmt.Fprintf(out, "%s AUDIT: id=%q ip=%q method=%q user=%q as=%q namespace=%q uri=%q\n",
			time.Now().Format(time.RFC3339Nano), ev.ID, utilnet.GetClientIP(req), req.Method, ev.User,
			ev.AsUser, ev.Namespace, req.URL)

		respWriter := constructResponseWriter(w, out, &ev) // catch response code
		handler.ServeHTTP(respWriter, req)

		fmt.Fprintf(out, "%s AUDIT: id=%q response=\"%d\" request=%v old=%v new=%v\n", time.Now().Format(time.RFC3339Nano), ev.ID, ev.Response, ev.RequestObject != nil, ev.OldObject != nil, ev.NewObject != nil)
	})
}

func constructResponseWriter(responseWriter http.ResponseWriter, out io.Writer, ev *genericaudit.Event) http.ResponseWriter {
	delegate := &auditResponseWriter{ResponseWriter: responseWriter, out: out, event: ev}
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
