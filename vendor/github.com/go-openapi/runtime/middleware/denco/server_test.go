package denco_test

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-openapi/runtime/middleware/denco"
)

func testHandlerFunc(w http.ResponseWriter, r *http.Request, params denco.Params) {
	fmt.Fprintf(w, "method: %s, path: %s, params: %v", r.Method, r.URL.Path, params)
}

func TestMux(t *testing.T) {
	mux := denco.NewMux()
	handler, err := mux.Build([]denco.Handler{
		mux.GET("/", testHandlerFunc),
		mux.GET("/user/:name", testHandlerFunc),
		mux.POST("/user/:name", testHandlerFunc),
		mux.HEAD("/user/:name", testHandlerFunc),
		mux.PUT("/user/:name", testHandlerFunc),
		mux.Handler("GET", "/user/handler", testHandlerFunc),
		mux.Handler("POST", "/user/handler", testHandlerFunc),
		{"PUT", "/user/inference", testHandlerFunc},
	})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(handler)
	defer server.Close()

	for _, v := range []struct {
		status                 int
		method, path, expected string
	}{
		{200, "GET", "/", "method: GET, path: /, params: []"},
		{200, "GET", "/user/alice", "method: GET, path: /user/alice, params: [{name alice}]"},
		{200, "POST", "/user/bob", "method: POST, path: /user/bob, params: [{name bob}]"},
		{200, "HEAD", "/user/alice", ""},
		{200, "PUT", "/user/bob", "method: PUT, path: /user/bob, params: [{name bob}]"},
		{404, "POST", "/", "404 page not found\n"},
		{404, "GET", "/unknown", "404 page not found\n"},
		{404, "POST", "/user/alice/1", "404 page not found\n"},
		{200, "GET", "/user/handler", "method: GET, path: /user/handler, params: []"},
		{200, "POST", "/user/handler", "method: POST, path: /user/handler, params: []"},
		{200, "PUT", "/user/inference", "method: PUT, path: /user/inference, params: []"},
	} {
		req, err := http.NewRequest(v.method, server.URL+v.path, nil)
		if err != nil {
			t.Error(err)
			continue
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Error(err)
			continue
		}
		defer res.Body.Close()
		body, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Error(err)
			continue
		}
		actual := string(body)
		expected := v.expected
		if res.StatusCode != v.status || actual != expected {
			t.Errorf(`%s "%s" => %#v %#v, want %#v %#v`, v.method, v.path, res.StatusCode, actual, v.status, expected)
		}
	}
}

func TestNotFound(t *testing.T) {
	mux := denco.NewMux()
	handler, err := mux.Build([]denco.Handler{})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(handler)
	defer server.Close()

	origNotFound := denco.NotFound
	defer func() {
		denco.NotFound = origNotFound
	}()
	denco.NotFound = func(w http.ResponseWriter, r *http.Request, params denco.Params) {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "method: %s, path: %s, params: %v", r.Method, r.URL.Path, params)
	}
	res, err := http.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	actual := string(body)
	expected := "method: GET, path: /, params: []"
	if res.StatusCode != http.StatusServiceUnavailable || actual != expected {
		t.Errorf(`GET "/" => %#v %#v, want %#v %#v`, res.StatusCode, actual, http.StatusServiceUnavailable, expected)
	}
}
