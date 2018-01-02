// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package logging

import (
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"

	"cloud.google.com/go/internal/testutil"
	"google.golang.org/api/option"
)

func TestLogPayload(t *testing.T) {
	lt := newLogTest(t)
	defer lt.ts.Close()

	tests := []struct {
		name  string
		entry Entry
		want  string
	}{
		{
			name: "string",
			entry: Entry{
				Time:    time.Unix(0, 0),
				Payload: "some log string",
			},
			want: `{"entries":[{"metadata":{"serviceName":"custom.googleapis.com","timestamp":"1970-01-01T00:00:00Z"},"textPayload":"some log string"}]}`,
		},
		{
			name: "[]byte",
			entry: Entry{
				Time:    time.Unix(0, 0),
				Payload: []byte("some log bytes"),
			},
			want: `{"entries":[{"metadata":{"serviceName":"custom.googleapis.com","timestamp":"1970-01-01T00:00:00Z"},"textPayload":"some log bytes"}]}`,
		},
		{
			name: "struct",
			entry: Entry{
				Time: time.Unix(0, 0),
				Payload: struct {
					Foo string `json:"foo"`
					Bar int    `json:"bar,omitempty"`
				}{
					Foo: "foovalue",
				},
			},
			want: `{"entries":[{"metadata":{"serviceName":"custom.googleapis.com","timestamp":"1970-01-01T00:00:00Z"},"structPayload":{"foo":"foovalue"}}]}`,
		},
		{
			name: "map[string]interface{}",
			entry: Entry{
				Time: time.Unix(0, 0),
				Payload: map[string]interface{}{
					"string": "foo",
					"int":    42,
				},
			},
			want: `{"entries":[{"metadata":{"serviceName":"custom.googleapis.com","timestamp":"1970-01-01T00:00:00Z"},"structPayload":{"int":42,"string":"foo"}}]}`,
		},
		{
			name: "map[string]interface{}",
			entry: Entry{
				Time:    time.Unix(0, 0),
				Payload: customJSONObject{},
			},
			want: `{"entries":[{"metadata":{"serviceName":"custom.googleapis.com","timestamp":"1970-01-01T00:00:00Z"},"structPayload":{"custom":"json"}}]}`,
		},
	}
	for _, tt := range tests {
		lt.startGetRequest()
		if err := lt.c.LogSync(tt.entry); err != nil {
			t.Errorf("%s: LogSync = %v", tt.name, err)
			continue
		}
		got := lt.getRequest()
		if got != tt.want {
			t.Errorf("%s: mismatch\n got: %s\nwant: %s\n", tt.name, got, tt.want)
		}
	}
}

func TestBufferInterval(t *testing.T) {
	lt := newLogTest(t)
	defer lt.ts.Close()

	lt.c.CommonLabels = map[string]string{
		"common1": "one",
		"common2": "two",
	}
	lt.c.BufferInterval = 1 * time.Millisecond // immediately, basically.
	lt.c.FlushAfter = 100                      // but we'll only send 1

	lt.startGetRequest()
	lt.c.Logger(Debug).Printf("log line 1")
	got := lt.getRequest()
	want := `{"commonLabels":{"common1":"one","common2":"two"},"entries":[{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:01Z"},"textPayload":"log line 1\n"}]}`
	if got != want {
		t.Errorf(" got: %s\nwant: %s\n", got, want)
	}
}

func TestFlushAfter(t *testing.T) {
	lt := newLogTest(t)
	defer lt.ts.Close()

	lt.c.CommonLabels = map[string]string{
		"common1": "one",
		"common2": "two",
	}
	lt.c.BufferInterval = getRequestTimeout * 2
	lt.c.FlushAfter = 2

	lt.c.Logger(Debug).Printf("log line 1")
	lt.startGetRequest()
	lt.c.Logger(Debug).Printf("log line 2")
	got := lt.getRequest()
	want := `{"commonLabels":{"common1":"one","common2":"two"},"entries":[{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:01Z"},"textPayload":"log line 1\n"},{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:02Z"},"textPayload":"log line 2\n"}]}`
	if got != want {
		t.Errorf(" got: %s\nwant: %s\n", got, want)
	}
}

func TestFlush(t *testing.T) {
	lt := newLogTest(t)
	defer lt.ts.Close()
	lt.c.BufferInterval = getRequestTimeout * 2
	lt.c.FlushAfter = 100 // but we'll only send 1, requiring a Flush

	lt.c.Logger(Debug).Printf("log line 1")
	lt.startGetRequest()
	if err := lt.c.Flush(); err != nil {
		t.Fatal(err)
	}
	got := lt.getRequest()
	want := `{"entries":[{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:01Z"},"textPayload":"log line 1\n"}]}`
	if got != want {
		t.Errorf(" got: %s\nwant: %s\n", got, want)
	}
}

func TestOverflow(t *testing.T) {
	lt := newLogTest(t)
	defer lt.ts.Close()

	lt.c.FlushAfter = 1
	lt.c.BufferLimit = 5
	lt.c.BufferInterval = 1 * time.Millisecond // immediately, basically.

	someErr := errors.New("some specific error value")
	lt.c.Overflow = func(c *Client, e Entry) error {
		return someErr
	}

	unblock := make(chan bool, 1)
	inHandler := make(chan bool, 1)
	lt.handlerc <- http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inHandler <- true
		<-unblock
		ioutil.ReadAll(r.Body)
		io.WriteString(w, "{}") // WriteLogEntriesResponse
	})

	lt.c.Logger(Debug).Printf("log line 1")
	<-inHandler
	lt.c.Logger(Debug).Printf("log line 2")
	lt.c.Logger(Debug).Printf("log line 3")
	lt.c.Logger(Debug).Printf("log line 4")
	lt.c.Logger(Debug).Printf("log line 5")

	queued, inFlight := lt.c.stats()
	if want := 4; queued != want {
		t.Errorf("queued = %d; want %d", queued, want)
	}
	if want := 1; inFlight != want {
		t.Errorf("inFlight = %d; want %d", inFlight, want)
	}

	if err := lt.c.Log(Entry{Payload: "to overflow"}); err != someErr {
		t.Errorf("Log(overflow Log entry) = %v; want someErr", err)
	}
	lt.startGetRequest()
	unblock <- true
	got := lt.getRequest()
	want := `{"entries":[{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:02Z"},"textPayload":"log line 2\n"},{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:03Z"},"textPayload":"log line 3\n"},{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:04Z"},"textPayload":"log line 4\n"},{"metadata":{"serviceName":"custom.googleapis.com","severity":"DEBUG","timestamp":"1970-01-01T00:00:05Z"},"textPayload":"log line 5\n"}]}`
	if got != want {
		t.Errorf(" got: %s\nwant: %s\n", got, want)
	}
	if err := lt.c.Flush(); err != nil {
		t.Fatal(err)
	}
	queued, inFlight = lt.c.stats()
	if want := 0; queued != want {
		t.Errorf("queued = %d; want %d", queued, want)
	}
	if want := 0; inFlight != want {
		t.Errorf("inFlight = %d; want %d", inFlight, want)
	}
}

func TestIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Integration tests skipped in short mode")
	}

	ctx := context.Background()
	ts := testutil.TokenSource(ctx, Scope)
	if ts == nil {
		t.Skip("Integration tests skipped. See CONTRIBUTING.md for details")
	}

	projID := testutil.ProjID()

	c, err := NewClient(ctx, projID, "logging-integration-test", option.WithTokenSource(ts))
	if err != nil {
		t.Fatalf("error creating client: %v", err)
	}

	if err := c.Ping(); err != nil {
		t.Fatalf("error pinging logging api: %v", err)
	}
	// Ping twice, to verify that deduping doesn't change the result.
	if err := c.Ping(); err != nil {
		t.Fatalf("error pinging logging api: %v", err)
	}

	if err := c.LogSync(Entry{Payload: customJSONObject{}}); err != nil {
		t.Fatalf("error writing log: %v", err)
	}

	if err := c.Log(Entry{Payload: customJSONObject{}}); err != nil {
		t.Fatalf("error writing log: %v", err)
	}

	if _, err := c.Writer(Default).Write([]byte("test log with io.Writer")); err != nil {
		t.Fatalf("error writing log using io.Writer: %v", err)
	}

	c.Logger(Default).Println("test log with log.Logger")

	if err := c.Flush(); err != nil {
		t.Fatalf("error flushing logs: %v", err)
	}
}

func TestIntegrationPingBadProject(t *testing.T) {
	if testing.Short() {
		t.Skip("Integration tests skipped in short mode")
	}

	ctx := context.Background()
	ts := testutil.TokenSource(ctx, Scope)
	if ts == nil {
		t.Skip("Integration tests skipped. See CONTRIBUTING.md for details")
	}

	for _, projID := range []string{
		testutil.ProjID() + "-BAD", // nonexistent project
		"amazing-height-519",       // exists, but wrong creds
	} {
		c, err := NewClient(ctx, projID, "logging-integration-test", option.WithTokenSource(ts))
		if err != nil {
			t.Fatalf("project %s: error creating client: %v", projID, err)
		}
		if err := c.Ping(); err == nil {
			t.Errorf("project %s: want error pinging logging api, got nil", projID)
		}
		// Ping twice, just to make sure the deduping doesn't mess with the result.
		if err := c.Ping(); err == nil {
			t.Errorf("project %s: want error pinging logging api, got nil", projID)
		}
	}
}

func (c *Client) stats() (queued, inFlight int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.queued), c.inFlight
}

type customJSONObject struct{}

func (customJSONObject) MarshalJSON() ([]byte, error) {
	return []byte(`{"custom":"json"}`), nil
}

type logTest struct {
	t        *testing.T
	ts       *httptest.Server
	c        *Client
	handlerc chan<- http.Handler

	bodyc chan string
}

func newLogTest(t *testing.T) *logTest {
	handlerc := make(chan http.Handler, 1)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case h := <-handlerc:
			h.ServeHTTP(w, r)
		default:
			slurp, _ := ioutil.ReadAll(r.Body)
			t.Errorf("Unexpected HTTP request received: %s", slurp)
			w.WriteHeader(500)
			io.WriteString(w, "unexpected HTTP request")
		}
	}))
	c, err := NewClient(context.Background(), "PROJ-ID", "LOG-NAME",
		option.WithEndpoint(ts.URL),
		option.WithTokenSource(dummyTokenSource{}), // prevent DefaultTokenSource
	)
	if err != nil {
		t.Fatal(err)
	}
	var clock struct {
		sync.Mutex
		now time.Time
	}
	c.timeNow = func() time.Time {
		clock.Lock()
		defer clock.Unlock()
		if clock.now.IsZero() {
			clock.now = time.Unix(0, 0)
		}
		clock.now = clock.now.Add(1 * time.Second)
		return clock.now
	}
	return &logTest{
		t:        t,
		ts:       ts,
		c:        c,
		handlerc: handlerc,
	}
}

func (lt *logTest) startGetRequest() {
	bodyc := make(chan string, 1)
	lt.bodyc = bodyc
	lt.handlerc <- http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		slurp, err := ioutil.ReadAll(r.Body)
		if err != nil {
			bodyc <- "ERROR: " + err.Error()
		} else {
			bodyc <- string(slurp)
		}
		io.WriteString(w, "{}") // a complete WriteLogEntriesResponse JSON struct
	})
}

const getRequestTimeout = 5 * time.Second

func (lt *logTest) getRequest() string {
	if lt.bodyc == nil {
		lt.t.Fatalf("getRequest called without previous startGetRequest")
	}
	select {
	case v := <-lt.bodyc:
		return strings.TrimSpace(v)
	case <-time.After(getRequestTimeout):
		lt.t.Fatalf("timeout waiting for request")
		panic("unreachable")
	}
}

// dummyTokenSource returns fake oauth2 tokens for local testing.
type dummyTokenSource struct{}

func (dummyTokenSource) Token() (*oauth2.Token, error) {
	return new(oauth2.Token), nil
}
