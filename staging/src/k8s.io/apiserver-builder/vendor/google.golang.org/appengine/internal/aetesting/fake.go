// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package aetesting provides utilities for testing App Engine packages.
// This is not for testing user applications.
package aetesting

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
)

// FakeSingleContext returns a context whose Call invocations will be serviced
// by f, which should be a function that has two arguments of the input and output
// protocol buffer type, and one error return.
func FakeSingleContext(t *testing.T, service, method string, f interface{}) context.Context {
	fv := reflect.ValueOf(f)
	if fv.Kind() != reflect.Func {
		t.Fatal("not a function")
	}
	ft := fv.Type()
	if ft.NumIn() != 2 || ft.NumOut() != 1 {
		t.Fatalf("f has %d in and %d out, want 2 in and 1 out", ft.NumIn(), ft.NumOut())
	}
	for i := 0; i < 2; i++ {
		at := ft.In(i)
		if !at.Implements(protoMessageType) {
			t.Fatalf("arg %d does not implement proto.Message", i)
		}
	}
	if ft.Out(0) != errorType {
		t.Fatalf("f's return is %v, want error", ft.Out(0))
	}
	s := &single{
		t:       t,
		service: service,
		method:  method,
		f:       fv,
	}
	return internal.WithCallOverride(internal.ContextForTesting(&http.Request{}), s.call)
}

var (
	protoMessageType = reflect.TypeOf((*proto.Message)(nil)).Elem()
	errorType        = reflect.TypeOf((*error)(nil)).Elem()
)

type single struct {
	t               *testing.T
	service, method string
	f               reflect.Value
}

func (s *single) call(ctx context.Context, service, method string, in, out proto.Message) error {
	if service == "__go__" {
		if method == "GetNamespace" {
			return nil // always yield an empty namespace
		}
		return fmt.Errorf("Unknown API call /%s.%s", service, method)
	}
	if service != s.service || method != s.method {
		s.t.Fatalf("Unexpected call to /%s.%s", service, method)
	}
	ins := []reflect.Value{
		reflect.ValueOf(in),
		reflect.ValueOf(out),
	}
	outs := s.f.Call(ins)
	if outs[0].IsNil() {
		return nil
	}
	return outs[0].Interface().(error)
}
