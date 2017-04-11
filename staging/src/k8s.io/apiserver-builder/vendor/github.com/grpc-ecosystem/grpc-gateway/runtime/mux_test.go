package runtime_test

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"github.com/grpc-ecosystem/grpc-gateway/utilities"
)

func TestMuxServeHTTP(t *testing.T) {
	type stubPattern struct {
		method string
		ops    []int
		pool   []string
		verb   string
	}
	for _, spec := range []struct {
		patterns []stubPattern

		reqMethod string
		reqPath   string
		headers   map[string]string

		respStatus  int
		respContent string
	}{
		{
			patterns:   nil,
			reqMethod:  "GET",
			reqPath:    "/",
			respStatus: http.StatusNotFound,
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
			},
			reqMethod:   "GET",
			reqPath:     "/foo",
			respStatus:  http.StatusOK,
			respContent: "GET /foo",
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
			},
			reqMethod:  "GET",
			reqPath:    "/bar",
			respStatus: http.StatusNotFound,
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
				{
					method: "GET",
					ops:    []int{int(utilities.OpPush), 0},
				},
			},
			reqMethod:   "GET",
			reqPath:     "/foo",
			respStatus:  http.StatusOK,
			respContent: "GET /foo",
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
				{
					method: "POST",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
			},
			reqMethod:   "POST",
			reqPath:     "/foo",
			respStatus:  http.StatusOK,
			respContent: "POST /foo",
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
			},
			reqMethod:  "DELETE",
			reqPath:    "/foo",
			respStatus: http.StatusMethodNotAllowed,
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
			},
			reqMethod: "POST",
			reqPath:   "/foo",
			headers: map[string]string{
				"Content-Type": "application/x-www-form-urlencoded",
			},
			respStatus:  http.StatusOK,
			respContent: "GET /foo",
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
				{
					method: "POST",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
			},
			reqMethod: "POST",
			reqPath:   "/foo",
			headers: map[string]string{
				"Content-Type":           "application/x-www-form-urlencoded",
				"X-HTTP-Method-Override": "GET",
			},
			respStatus:  http.StatusOK,
			respContent: "GET /foo",
		},
		{
			patterns: []stubPattern{
				{
					method: "GET",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
				},
			},
			reqMethod: "POST",
			reqPath:   "/foo",
			headers: map[string]string{
				"Content-Type": "application/json",
			},
			respStatus: http.StatusMethodNotAllowed,
		},
		{
			patterns: []stubPattern{
				{
					method: "POST",
					ops:    []int{int(utilities.OpLitPush), 0},
					pool:   []string{"foo"},
					verb:   "bar",
				},
			},
			reqMethod: "POST",
			reqPath:   "/foo:bar",
			headers: map[string]string{
				"Content-Type": "application/json",
			},
			respStatus:  http.StatusOK,
			respContent: "POST /foo:bar",
		},
	} {
		mux := runtime.NewServeMux()
		for _, p := range spec.patterns {
			func(p stubPattern) {
				pat, err := runtime.NewPattern(1, p.ops, p.pool, p.verb)
				if err != nil {
					t.Fatalf("runtime.NewPattern(1, %#v, %#v, %q) failed with %v; want success", p.ops, p.pool, p.verb, err)
				}
				mux.Handle(p.method, pat, func(w http.ResponseWriter, r *http.Request, pathParams map[string]string) {
					fmt.Fprintf(w, "%s %s", p.method, pat.String())
				})
			}(p)
		}

		url := fmt.Sprintf("http://host.example%s", spec.reqPath)
		r, err := http.NewRequest(spec.reqMethod, url, bytes.NewReader(nil))
		if err != nil {
			t.Fatalf("http.NewRequest(%q, %q, nil) failed with %v; want success", spec.reqMethod, url, err)
		}
		for name, value := range spec.headers {
			r.Header.Set(name, value)
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, r)

		if got, want := w.Code, spec.respStatus; got != want {
			t.Errorf("w.Code = %d; want %d; patterns=%v; req=%v", got, want, spec.patterns, r)
		}
		if spec.respContent != "" {
			if got, want := w.Body.String(), spec.respContent; got != want {
				t.Errorf("w.Body = %q; want %q; patterns=%v; req=%v", got, want, spec.patterns, r)
			}
		}
	}
}
