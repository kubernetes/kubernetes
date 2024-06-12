// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

//go:build appengine
// +build appengine

package internal

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"appengine"
	"appengine_internal"
	basepb "appengine_internal/base"

	"github.com/golang/protobuf/proto"
)

var contextKey = "holds an appengine.Context"

// fromContext returns the App Engine context or nil if ctx is not
// derived from an App Engine context.
func fromContext(ctx context.Context) appengine.Context {
	c, _ := ctx.Value(&contextKey).(appengine.Context)
	return c
}

// This is only for classic App Engine adapters.
func ClassicContextFromContext(ctx context.Context) (appengine.Context, error) {
	c := fromContext(ctx)
	if c == nil {
		return nil, errNotAppEngineContext
	}
	return c, nil
}

func withContext(parent context.Context, c appengine.Context) context.Context {
	ctx := context.WithValue(parent, &contextKey, c)

	s := &basepb.StringProto{}
	c.Call("__go__", "GetNamespace", &basepb.VoidProto{}, s, nil)
	if ns := s.GetValue(); ns != "" {
		ctx = NamespacedContext(ctx, ns)
	}

	return ctx
}

func IncomingHeaders(ctx context.Context) http.Header {
	if c := fromContext(ctx); c != nil {
		if req, ok := c.Request().(*http.Request); ok {
			return req.Header
		}
	}
	return nil
}

func ReqContext(req *http.Request) context.Context {
	return WithContext(context.Background(), req)
}

func WithContext(parent context.Context, req *http.Request) context.Context {
	c := appengine.NewContext(req)
	return withContext(parent, c)
}

type testingContext struct {
	appengine.Context

	req *http.Request
}

func (t *testingContext) FullyQualifiedAppID() string { return "dev~testcontext" }
func (t *testingContext) Call(service, method string, _, _ appengine_internal.ProtoMessage, _ *appengine_internal.CallOptions) error {
	if service == "__go__" && method == "GetNamespace" {
		return nil
	}
	return fmt.Errorf("testingContext: unsupported Call")
}
func (t *testingContext) Request() interface{} { return t.req }

func ContextForTesting(req *http.Request) context.Context {
	return withContext(context.Background(), &testingContext{req: req})
}

func Call(ctx context.Context, service, method string, in, out proto.Message) error {
	if ns := NamespaceFromContext(ctx); ns != "" {
		if fn, ok := NamespaceMods[service]; ok {
			fn(in, ns)
		}
	}

	if f, ctx, ok := callOverrideFromContext(ctx); ok {
		return f(ctx, service, method, in, out)
	}

	// Handle already-done contexts quickly.
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	c := fromContext(ctx)
	if c == nil {
		// Give a good error message rather than a panic lower down.
		return errNotAppEngineContext
	}

	// Apply transaction modifications if we're in a transaction.
	if t := transactionFromContext(ctx); t != nil {
		if t.finished {
			return errors.New("transaction context has expired")
		}
		applyTransaction(in, &t.transaction)
	}

	var opts *appengine_internal.CallOptions
	if d, ok := ctx.Deadline(); ok {
		opts = &appengine_internal.CallOptions{
			Timeout: d.Sub(time.Now()),
		}
	}

	err := c.Call(service, method, in, out, opts)
	switch v := err.(type) {
	case *appengine_internal.APIError:
		return &APIError{
			Service: v.Service,
			Detail:  v.Detail,
			Code:    v.Code,
		}
	case *appengine_internal.CallError:
		return &CallError{
			Detail:  v.Detail,
			Code:    v.Code,
			Timeout: v.Timeout,
		}
	}
	return err
}

func Middleware(next http.Handler) http.Handler {
	panic("Middleware called; this should be impossible")
}

func logf(c appengine.Context, level int64, format string, args ...interface{}) {
	var fn func(format string, args ...interface{})
	switch level {
	case 0:
		fn = c.Debugf
	case 1:
		fn = c.Infof
	case 2:
		fn = c.Warningf
	case 3:
		fn = c.Errorf
	case 4:
		fn = c.Criticalf
	default:
		// This shouldn't happen.
		fn = c.Criticalf
	}
	fn(format, args...)
}
