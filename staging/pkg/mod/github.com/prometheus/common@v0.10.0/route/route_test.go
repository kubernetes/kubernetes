// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package route

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRedirect(t *testing.T) {
	router := New().WithPrefix("/test/prefix")
	w := httptest.NewRecorder()
	r, err := http.NewRequest("GET", "http://localhost:9090/foo", nil)
	if err != nil {
		t.Fatalf("Error building test request: %s", err)
	}

	router.Redirect(w, r, "/some/endpoint", http.StatusFound)
	if w.Code != http.StatusFound {
		t.Fatalf("Unexpected redirect status code: got %d, want %d", w.Code, http.StatusFound)
	}

	want := "/test/prefix/some/endpoint"
	got := w.Header()["Location"][0]
	if want != got {
		t.Fatalf("Unexpected redirect location: got %s, want %s", got, want)
	}
}

func TestContext(t *testing.T) {
	router := New()
	router.Get("/test/:foo/", func(w http.ResponseWriter, r *http.Request) {
		want := "bar"
		got := Param(r.Context(), "foo")
		if want != got {
			t.Fatalf("Unexpected context value: want %q, got %q", want, got)
		}
	})

	r, err := http.NewRequest("GET", "http://localhost:9090/test/bar/", nil)
	if err != nil {
		t.Fatalf("Error building test request: %s", err)
	}
	router.ServeHTTP(nil, r)
}

func TestContextWithValue(t *testing.T) {
	router := New()
	router.Get("/test/:foo/", func(w http.ResponseWriter, r *http.Request) {
		want := "bar"
		got := Param(r.Context(), "foo")
		if want != got {
			t.Fatalf("Unexpected context value: want %q, got %q", want, got)
		}
		want = "ipsum"
		got = Param(r.Context(), "lorem")
		if want != got {
			t.Fatalf("Unexpected context value: want %q, got %q", want, got)
		}
		want = "sit"
		got = Param(r.Context(), "dolor")
		if want != got {
			t.Fatalf("Unexpected context value: want %q, got %q", want, got)
		}
	})

	r, err := http.NewRequest("GET", "http://localhost:9090/test/bar/", nil)
	if err != nil {
		t.Fatalf("Error building test request: %s", err)
	}
	params := map[string]string{
		"lorem": "ipsum",
		"dolor": "sit",
	}

	ctx := r.Context()
	for p, v := range params {
		ctx = WithParam(ctx, p, v)
	}
	r = r.WithContext(ctx)
	router.ServeHTTP(nil, r)
}

func TestContextWithoutValue(t *testing.T) {
	router := New()
	router.Get("/test", func(w http.ResponseWriter, r *http.Request) {
		want := ""
		got := Param(r.Context(), "foo")
		if want != got {
			t.Fatalf("Unexpected context value: want %q, got %q", want, got)
		}
	})

	r, err := http.NewRequest("GET", "http://localhost:9090/test", nil)
	if err != nil {
		t.Fatalf("Error building test request: %s", err)
	}
	router.ServeHTTP(nil, r)
}

func TestInstrumentation(t *testing.T) {
	var got string
	cases := []struct {
		router *Router
		want   string
	}{
		{
			router: New(),
			want:   "",
		}, {
			router: New().WithInstrumentation(func(handlerName string, handler http.HandlerFunc) http.HandlerFunc {
				got = handlerName
				return handler
			}),
			want: "/foo",
		},
	}

	for _, c := range cases {
		c.router.Get("/foo", func(w http.ResponseWriter, r *http.Request) {})

		r, err := http.NewRequest("GET", "http://localhost:9090/foo", nil)
		if err != nil {
			t.Fatalf("Error building test request: %s", err)
		}
		c.router.ServeHTTP(nil, r)
		if c.want != got {
			t.Fatalf("Unexpected value: want %q, got %q", c.want, got)
		}
	}
}

func TestInstrumentations(t *testing.T) {
	got := make([]string, 0)
	cases := []struct {
		router *Router
		want   []string
	}{
		{
			router: New(),
			want:   []string{},
		}, {
			router: New().
				WithInstrumentation(
					func(handlerName string, handler http.HandlerFunc) http.HandlerFunc {
						got = append(got, "1"+handlerName)
						return handler
					}).
				WithInstrumentation(
					func(handlerName string, handler http.HandlerFunc) http.HandlerFunc {
						got = append(got, "2"+handlerName)
						return handler
					}).
				WithInstrumentation(
					func(handlerName string, handler http.HandlerFunc) http.HandlerFunc {
						got = append(got, "3"+handlerName)
						return handler
					}),
			want: []string{"1/foo", "2/foo", "3/foo"},
		},
	}

	for _, c := range cases {
		c.router.Get("/foo", func(w http.ResponseWriter, r *http.Request) {})

		r, err := http.NewRequest("GET", "http://localhost:9090/foo", nil)
		if err != nil {
			t.Fatalf("Error building test request: %s", err)
		}
		c.router.ServeHTTP(nil, r)
		if len(c.want) != len(got) {
			t.Fatalf("Unexpected value: want %q, got %q", c.want, got)
		}
		for i, v := range c.want {
			if v != got[i] {
				t.Fatalf("Unexpected value: want %q, got %q", c.want, got)
			}
		}
	}
}
