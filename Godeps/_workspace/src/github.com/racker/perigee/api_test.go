package perigee

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNormal(t *testing.T) {
	handler := http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte("testing"))
		})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	response, err := Request("GET", ts.URL, Options{})
	if err != nil {
		t.Fatalf("should not have error: %s", err)
	}
	if response.StatusCode != 200 {
		t.Fatalf("response code %d is not 200", response.StatusCode)
	}
}

func TestOKCodes(t *testing.T) {
	expectCode := 201
	handler := http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(expectCode)
			w.Write([]byte("testing"))
		})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	options := Options{
		OkCodes: []int{expectCode},
	}
	results, err := Request("GET", ts.URL, options)
	if err != nil {
		t.Fatalf("should not have error: %s", err)
	}
	if results.StatusCode != expectCode {
		t.Fatalf("response code %d is not %d", results.StatusCode, expectCode)
	}
}

func TestLocation(t *testing.T) {
	newLocation := "http://www.example.com"
	handler := http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Location", newLocation)
			w.Write([]byte("testing"))
		})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	response, err := Request("GET", ts.URL, Options{})
	if err != nil {
		t.Fatalf("should not have error: %s", err)
	}

	location, err := response.HttpResponse.Location()
	if err != nil {
		t.Fatalf("should not have error: %s", err)
	}

	if location.String() != newLocation {
		t.Fatalf("location returned \"%s\" is not \"%s\"", location.String(), newLocation)
	}
}

func TestHeaders(t *testing.T) {
	newLocation := "http://www.example.com"
	handler := http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Location", newLocation)
			w.Write([]byte("testing"))
		})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	response, err := Request("GET", ts.URL, Options{})
	if err != nil {
		t.Fatalf("should not have error: %s", err)
	}

	location := response.HttpResponse.Header.Get("Location")
	if location == "" {
		t.Fatalf("Location should not empty")
	}

	if location != newLocation {
		t.Fatalf("location returned \"%s\" is not \"%s\"", location, newLocation)
	}
}

func TestCustomHeaders(t *testing.T) {
	var contentType, accept, contentLength string

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m := map[string][]string(r.Header)
		contentType = m["Content-Type"][0]
		accept = m["Accept"][0]
		contentLength = m["Content-Length"][0]
	})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	_, err := Request("GET", ts.URL, Options{
		ContentLength: 5,
		ContentType:   "x-application/vb",
		Accept:        "x-application/c",
		ReqBody:       strings.NewReader("Hello"),
	})
	if err != nil {
		t.Fatalf(err.Error())
	}

	if contentType != "x-application/vb" {
		t.Fatalf("I expected x-application/vb; got ", contentType)
	}

	if contentLength != "5" {
		t.Fatalf("I expected 5 byte content length; got ", contentLength)
	}

	if accept != "x-application/c" {
		t.Fatalf("I expected x-application/c; got ", accept)
	}
}

func TestJson(t *testing.T) {
	newLocation := "http://www.example.com"
	jsonBytes := []byte(`{"foo": {"bar": "baz"}}`)
	handler := http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Location", newLocation)
			w.Write(jsonBytes)
		})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	type Data struct {
		Foo struct {
			Bar string `json:"bar"`
		} `json:"foo"`
	}
	var data Data

	response, err := Request("GET", ts.URL, Options{Results: &data})
	if err != nil {
		t.Fatalf("should not have error: %s", err)
	}

	if bytes.Compare(jsonBytes, response.JsonResult) != 0 {
		t.Fatalf("json returned \"%s\" is not \"%s\"", response.JsonResult, jsonBytes)
	}

	if data.Foo.Bar != "baz" {
		t.Fatalf("Results returned %v", data)
	}
}

func TestSetHeaders(t *testing.T) {
	var wasCalled bool
	handler := http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte("Hi"))
		})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	_, err := Request("GET", ts.URL, Options{
		SetHeaders: func(r *http.Request) error {
			wasCalled = true
			return nil
		},
	})

	if err != nil {
		t.Fatal(err)
	}

	if !wasCalled {
		t.Fatal("I expected header setter callback to be called, but it wasn't")
	}

	myError := fmt.Errorf("boo")

	_, err = Request("GET", ts.URL, Options{
		SetHeaders: func(r *http.Request) error {
			return myError
		},
	})

	if err != myError {
		t.Fatal("I expected errors to propegate back to the caller.")
	}
}

func TestBodilessMethodsAreSentWithoutContentHeaders(t *testing.T) {
	var h map[string][]string

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h = r.Header
	})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	_, err := Request("GET", ts.URL, Options{})
	if err != nil {
		t.Fatalf(err.Error())
	}

	if len(h["Content-Type"]) != 0 {
		t.Fatalf("I expected nothing for Content-Type but got ", h["Content-Type"])
	}

	if len(h["Content-Length"]) != 0 {
		t.Fatalf("I expected nothing for Content-Length but got ", h["Content-Length"])
	}
}
