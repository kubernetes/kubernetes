// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestNewAPIClient(t *testing.T) {
	endpoint := "http://localhost:4243"
	client, err := NewClient(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if client.endpoint != endpoint {
		t.Errorf("Expected endpoint %s. Got %s.", endpoint, client.endpoint)
	}
	if client.HTTPClient != http.DefaultClient {
		t.Errorf("Expected http.Client %#v. Got %#v.", http.DefaultClient, client.HTTPClient)
	}
	// test unix socket endpoints
	endpoint = "unix:///var/run/docker.sock"
	client, err = NewClient(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if client.endpoint != endpoint {
		t.Errorf("Expected endpoint %s. Got %s.", endpoint, client.endpoint)
	}
	if !client.SkipServerVersionCheck {
		t.Error("Expected SkipServerVersionCheck to be true, got false")
	}
	if client.requestedApiVersion != nil {
		t.Errorf("Expected requestedApiVersion to be nil, got %#v.", client.requestedApiVersion)
	}
}

func TestNewVersionedClient(t *testing.T) {
	endpoint := "http://localhost:4243"
	client, err := NewVersionedClient(endpoint, "1.12")
	if err != nil {
		t.Fatal(err)
	}
	if client.endpoint != endpoint {
		t.Errorf("Expected endpoint %s. Got %s.", endpoint, client.endpoint)
	}
	if client.HTTPClient != http.DefaultClient {
		t.Errorf("Expected http.Client %#v. Got %#v.", http.DefaultClient, client.HTTPClient)
	}
	if reqVersion := client.requestedApiVersion.String(); reqVersion != "1.12" {
		t.Errorf("Wrong requestApiVersion. Want %q. Got %q.", "1.12", reqVersion)
	}
	if client.SkipServerVersionCheck {
		t.Error("Expected SkipServerVersionCheck to be false, got true")
	}
}

func TestNewClientInvalidEndpoint(t *testing.T) {
	cases := []string{
		"htp://localhost:3243", "http://localhost:a", "localhost:8080",
		"", "localhost", "http://localhost:8080:8383", "http://localhost:65536",
		"https://localhost:-20",
	}
	for _, c := range cases {
		client, err := NewClient(c)
		if client != nil {
			t.Errorf("Want <nil> client for invalid endpoint, got %#v.", client)
		}
		if !reflect.DeepEqual(err, ErrInvalidEndpoint) {
			t.Errorf("NewClient(%q): Got invalid error for invalid endpoint. Want %#v. Got %#v.", c, ErrInvalidEndpoint, err)
		}
	}
}

func TestGetURL(t *testing.T) {
	var tests = []struct {
		endpoint string
		path     string
		expected string
	}{
		{"http://localhost:4243/", "/", "http://localhost:4243/"},
		{"http://localhost:4243", "/", "http://localhost:4243/"},
		{"http://localhost:4243", "/containers/ps", "http://localhost:4243/containers/ps"},
		{"tcp://localhost:4243", "/containers/ps", "http://localhost:4243/containers/ps"},
		{"http://localhost:4243/////", "/", "http://localhost:4243/"},
		{"unix:///var/run/docker.socket", "/containers", "/containers"},
	}
	for _, tt := range tests {
		client, _ := NewClient(tt.endpoint)
		client.endpoint = tt.endpoint
		client.SkipServerVersionCheck = true
		got := client.getURL(tt.path)
		if got != tt.expected {
			t.Errorf("getURL(%q): Got %s. Want %s.", tt.path, got, tt.expected)
		}
	}
}

func TestError(t *testing.T) {
	err := newError(400, []byte("bad parameter"))
	expected := Error{Status: 400, Message: "bad parameter"}
	if !reflect.DeepEqual(expected, *err) {
		t.Errorf("Wrong error type. Want %#v. Got %#v.", expected, *err)
	}
	message := "API error (400): bad parameter"
	if err.Error() != message {
		t.Errorf("Wrong error message. Want %q. Got %q.", message, err.Error())
	}
}

func TestQueryString(t *testing.T) {
	v := float32(2.4)
	f32QueryString := fmt.Sprintf("w=%s&x=10&y=10.35", strconv.FormatFloat(float64(v), 'f', -1, 64))
	jsonPerson := url.QueryEscape(`{"Name":"gopher","age":4}`)
	var tests = []struct {
		input interface{}
		want  string
	}{
		{&ListContainersOptions{All: true}, "all=1"},
		{ListContainersOptions{All: true}, "all=1"},
		{ListContainersOptions{Before: "something"}, "before=something"},
		{ListContainersOptions{Before: "something", Since: "other"}, "before=something&since=other"},
		{dumb{X: 10, Y: 10.35000}, "x=10&y=10.35"},
		{dumb{W: v, X: 10, Y: 10.35000}, f32QueryString},
		{dumb{X: 10, Y: 10.35000, Z: 10}, "x=10&y=10.35&zee=10"},
		{dumb{v: 4, X: 10, Y: 10.35000}, "x=10&y=10.35"},
		{dumb{T: 10, Y: 10.35000}, "y=10.35"},
		{dumb{Person: &person{Name: "gopher", Age: 4}}, "p=" + jsonPerson},
		{nil, ""},
		{10, ""},
		{"not_a_struct", ""},
	}
	for _, tt := range tests {
		got := queryString(tt.input)
		if got != tt.want {
			t.Errorf("queryString(%v). Want %q. Got %q.", tt.input, tt.want, got)
		}
	}
}

func TestNewApiVersionFailures(t *testing.T) {
	var tests = []struct {
		input         string
		expectedError string
	}{
		{"1-0", `Unable to parse version "1-0"`},
		{"1.0-beta", `Unable to parse version "1.0-beta": "0-beta" is not an integer`},
	}
	for _, tt := range tests {
		v, err := NewApiVersion(tt.input)
		if v != nil {
			t.Errorf("Expected <nil> version, got %v.", v)
		}
		if err.Error() != tt.expectedError {
			t.Errorf("NewApiVersion(%q): wrong error. Want %q. Got %q", tt.input, tt.expectedError, err.Error())
		}
	}
}

func TestApiVersions(t *testing.T) {
	var tests = []struct {
		a                              string
		b                              string
		expectedALessThanB             bool
		expectedALessThanOrEqualToB    bool
		expectedAGreaterThanB          bool
		expectedAGreaterThanOrEqualToB bool
	}{
		{"1.11", "1.11", false, true, false, true},
		{"1.10", "1.11", true, true, false, false},
		{"1.11", "1.10", false, false, true, true},

		{"1.9", "1.11", true, true, false, false},
		{"1.11", "1.9", false, false, true, true},

		{"1.1.1", "1.1", false, false, true, true},
		{"1.1", "1.1.1", true, true, false, false},

		{"2.1", "1.1.1", false, false, true, true},
		{"2.1", "1.3.1", false, false, true, true},
		{"1.1.1", "2.1", true, true, false, false},
		{"1.3.1", "2.1", true, true, false, false},
	}

	for _, tt := range tests {
		a, _ := NewApiVersion(tt.a)
		b, _ := NewApiVersion(tt.b)

		if tt.expectedALessThanB && !a.LessThan(b) {
			t.Errorf("Expected %#v < %#v", a, b)
		}
		if tt.expectedALessThanOrEqualToB && !a.LessThanOrEqualTo(b) {
			t.Errorf("Expected %#v <= %#v", a, b)
		}
		if tt.expectedAGreaterThanB && !a.GreaterThan(b) {
			t.Errorf("Expected %#v > %#v", a, b)
		}
		if tt.expectedAGreaterThanOrEqualToB && !a.GreaterThanOrEqualTo(b) {
			t.Errorf("Expected %#v >= %#v", a, b)
		}
	}
}

func TestPing(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusOK}
	client := newTestClient(fakeRT)
	err := client.Ping()
	if err != nil {
		t.Fatal(err)
	}
}

func TestPingFailing(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusInternalServerError}
	client := newTestClient(fakeRT)
	err := client.Ping()
	if err == nil {
		t.Fatal("Expected non nil error, got nil")
	}
	expectedErrMsg := "API error (500): "
	if err.Error() != expectedErrMsg {
		t.Fatalf("Expected error to be %q, got: %q", expectedErrMsg, err.Error())
	}
}

func TestPingFailingWrongStatus(t *testing.T) {
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusAccepted}
	client := newTestClient(fakeRT)
	err := client.Ping()
	if err == nil {
		t.Fatal("Expected non nil error, got nil")
	}
	expectedErrMsg := "API error (202): "
	if err.Error() != expectedErrMsg {
		t.Fatalf("Expected error to be %q, got: %q", expectedErrMsg, err.Error())
	}
}

type FakeRoundTripper struct {
	message  string
	status   int
	requests []*http.Request
}

func (rt *FakeRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	body := strings.NewReader(rt.message)
	rt.requests = append(rt.requests, r)
	return &http.Response{
		StatusCode: rt.status,
		Body:       ioutil.NopCloser(body),
	}, nil
}

func (rt *FakeRoundTripper) Reset() {
	rt.requests = nil
}

type person struct {
	Name string
	Age  int `json:"age"`
}

type dumb struct {
	T      int `qs:"-"`
	v      int
	W      float32
	X      int
	Y      float64
	Z      int     `qs:"zee"`
	Person *person `qs:"p"`
}

type fakeEndpointURL struct {
	Scheme string
}
