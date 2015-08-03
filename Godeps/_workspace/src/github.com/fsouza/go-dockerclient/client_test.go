// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
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
	if client.requestedAPIVersion != nil {
		t.Errorf("Expected requestedAPIVersion to be nil, got %#v.", client.requestedAPIVersion)
	}
}

func newTLSClient(endpoint string) (*Client, error) {
	return NewTLSClient(endpoint,
		"testing/data/cert.pem",
		"testing/data/key.pem",
		"testing/data/ca.pem")
}

func TestNewTSLAPIClient(t *testing.T) {
	endpoint := "https://localhost:4243"
	client, err := newTLSClient(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if client.endpoint != endpoint {
		t.Errorf("Expected endpoint %s. Got %s.", endpoint, client.endpoint)
	}
	if !client.SkipServerVersionCheck {
		t.Error("Expected SkipServerVersionCheck to be true, got false")
	}
	if client.requestedAPIVersion != nil {
		t.Errorf("Expected requestedAPIVersion to be nil, got %#v.", client.requestedAPIVersion)
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
	if reqVersion := client.requestedAPIVersion.String(); reqVersion != "1.12" {
		t.Errorf("Wrong requestAPIVersion. Want %q. Got %q.", "1.12", reqVersion)
	}
	if client.SkipServerVersionCheck {
		t.Error("Expected SkipServerVersionCheck to be false, got true")
	}
}

func TestNewTLSVersionedClient(t *testing.T) {
	certPath := "testing/data/cert.pem"
	keyPath := "testing/data/key.pem"
	caPath := "testing/data/ca.pem"
	endpoint := "https://localhost:4243"
	client, err := NewVersionedTLSClient(endpoint, certPath, keyPath, caPath, "1.14")
	if err != nil {
		t.Fatal(err)
	}
	if client.endpoint != endpoint {
		t.Errorf("Expected endpoint %s. Got %s.", endpoint, client.endpoint)
	}
	if reqVersion := client.requestedAPIVersion.String(); reqVersion != "1.14" {
		t.Errorf("Wrong requestAPIVersion. Want %q. Got %q.", "1.14", reqVersion)
	}
	if client.SkipServerVersionCheck {
		t.Error("Expected SkipServerVersionCheck to be false, got true")
	}
}

func TestNewTLSVersionedClientInvalidCA(t *testing.T) {
	certPath := "testing/data/cert.pem"
	keyPath := "testing/data/key.pem"
	caPath := "testing/data/key.pem"
	endpoint := "https://localhost:4243"
	_, err := NewVersionedTLSClient(endpoint, certPath, keyPath, caPath, "1.14")
	if err == nil {
		t.Errorf("Expected invalid ca at %s", caPath)
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

func TestNewTLSClient(t *testing.T) {
	var tests = []struct {
		endpoint string
		expected string
	}{
		{"tcp://localhost:2376", "https"},
		{"tcp://localhost:2375", "https"},
		{"tcp://localhost:4000", "https"},
		{"http://localhost:4000", "https"},
	}

	for _, tt := range tests {
		client, err := newTLSClient(tt.endpoint)
		if err != nil {
			t.Error(err)
		}
		got := client.endpointURL.Scheme
		if got != tt.expected {
			t.Errorf("endpointURL.Scheme: Got %s. Want %s.", got, tt.expected)
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
		{ListContainersOptions{Filters: map[string][]string{"status": {"paused", "running"}}}, "filters=%7B%22status%22%3A%5B%22paused%22%2C%22running%22%5D%7D"},
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

func TestNewAPIVersionFailures(t *testing.T) {
	var tests = []struct {
		input         string
		expectedError string
	}{
		{"1-0", `Unable to parse version "1-0"`},
		{"1.0-beta", `Unable to parse version "1.0-beta": "0-beta" is not an integer`},
	}
	for _, tt := range tests {
		v, err := NewAPIVersion(tt.input)
		if v != nil {
			t.Errorf("Expected <nil> version, got %v.", v)
		}
		if err.Error() != tt.expectedError {
			t.Errorf("NewAPIVersion(%q): wrong error. Want %q. Got %q", tt.input, tt.expectedError, err.Error())
		}
	}
}

func TestAPIVersions(t *testing.T) {
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
		a, _ := NewAPIVersion(tt.a)
		b, _ := NewAPIVersion(tt.b)

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

func TestPingErrorWithUnixSocket(t *testing.T) {
	go func() {
		li, err := net.Listen("unix", "/tmp/echo.sock")
		defer li.Close()
		if err != nil {
			t.Fatalf("Expected to get listner, but failed: %#v", err)
			return
		}

		fd, err := li.Accept()
		if err != nil {
			t.Fatalf("Expected to accept connection, but failed: %#v", err)
			return
		}

		buf := make([]byte, 512)
		nr, err := fd.Read(buf)

		// Create invalid response message to occur error
		data := buf[0:nr]
		for i := 0; i < 10; i++ {
			data[i] = 63
		}

		_, err = fd.Write(data)
		if err != nil {
			t.Fatalf("Expected to write to socket, but failed: %#v", err)
		}

		return
	}()

	// Wait for unix socket to listen
	time.Sleep(10 * time.Millisecond)

	endpoint := "unix:///tmp/echo.sock"
	u, _ := parseEndpoint(endpoint, false)
	client := Client{
		HTTPClient:             http.DefaultClient,
		endpoint:               endpoint,
		endpointURL:            u,
		SkipServerVersionCheck: true,
	}

	err := client.Ping()
	if err == nil {
		t.Fatal("Expected non nil error, got nil")
	}
}

type FakeRoundTripper struct {
	message  string
	status   int
	header   map[string]string
	requests []*http.Request
}

func (rt *FakeRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	body := strings.NewReader(rt.message)
	rt.requests = append(rt.requests, r)
	res := &http.Response{
		StatusCode: rt.status,
		Body:       ioutil.NopCloser(body),
		Header:     make(http.Header),
	}
	for k, v := range rt.header {
		res.Header.Set(k, v)
	}
	return res, nil
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
