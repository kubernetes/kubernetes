package plugins

import (
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/docker/docker/pkg/tlsconfig"
)

var (
	mux    *http.ServeMux
	server *httptest.Server
)

func setupRemotePluginServer() string {
	mux = http.NewServeMux()
	server = httptest.NewServer(mux)
	return server.URL
}

func teardownRemotePluginServer() {
	if server != nil {
		server.Close()
	}
}

func TestFailedConnection(t *testing.T) {
	c, _ := NewClient("tcp://127.0.0.1:1", tlsconfig.Options{InsecureSkipVerify: true})
	err := c.callWithRetry("Service.Method", nil, nil, false)
	if err == nil {
		t.Fatal("Unexpected successful connection")
	}
}

func TestEchoInputOutput(t *testing.T) {
	addr := setupRemotePluginServer()
	defer teardownRemotePluginServer()

	m := Manifest{[]string{"VolumeDriver", "NetworkDriver"}}

	mux.HandleFunc("/Test.Echo", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Fatalf("Expected POST, got %s\n", r.Method)
		}

		header := w.Header()
		header.Set("Content-Type", versionMimetype)

		io.Copy(w, r.Body)
	})

	c, _ := NewClient(addr, tlsconfig.Options{InsecureSkipVerify: true})
	var output Manifest
	err := c.Call("Test.Echo", m, &output)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(output, m) {
		t.Fatalf("Expected %v, was %v\n", m, output)
	}
}

func TestBackoff(t *testing.T) {
	cases := []struct {
		retries    int
		expTimeOff time.Duration
	}{
		{0, time.Duration(1)},
		{1, time.Duration(2)},
		{2, time.Duration(4)},
		{4, time.Duration(16)},
		{6, time.Duration(30)},
		{10, time.Duration(30)},
	}

	for _, c := range cases {
		s := c.expTimeOff * time.Second
		if d := backoff(c.retries); d != s {
			t.Fatalf("Retry %v, expected %v, was %v\n", c.retries, s, d)
		}
	}
}

func TestAbortRetry(t *testing.T) {
	cases := []struct {
		timeOff  time.Duration
		expAbort bool
	}{
		{time.Duration(1), false},
		{time.Duration(2), false},
		{time.Duration(10), false},
		{time.Duration(30), true},
		{time.Duration(40), true},
	}

	for _, c := range cases {
		s := c.timeOff * time.Second
		if a := abort(time.Now(), s); a != c.expAbort {
			t.Fatalf("Duration %v, expected %v, was %v\n", c.timeOff, s, a)
		}
	}
}
