// Copyright 2016 Google LLC
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

package errorreporting

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/internal/testutil"
	gax "github.com/googleapis/gax-go/v2"
	"google.golang.org/api/option"
	pb "google.golang.org/genproto/googleapis/devtools/clouderrorreporting/v1beta1"
)

type fakeReportErrorsClient struct {
	req    *pb.ReportErrorEventRequest
	fail   bool
	doneCh chan struct{}
}

func (c *fakeReportErrorsClient) ReportErrorEvent(ctx context.Context, req *pb.ReportErrorEventRequest, _ ...gax.CallOption) (*pb.ReportErrorEventResponse, error) {
	defer close(c.doneCh)
	if c.fail {
		return nil, errors.New("request failed")
	}
	c.req = req
	return &pb.ReportErrorEventResponse{}, nil
}

func (c *fakeReportErrorsClient) Close() error {
	return nil
}

var defaultConfig = Config{
	ServiceName:    "myservice",
	ServiceVersion: "v1.0",
}

func newFakeReportErrorsClient() *fakeReportErrorsClient {
	c := &fakeReportErrorsClient{}
	c.doneCh = make(chan struct{})
	return c
}

func newTestClient(c *fakeReportErrorsClient, cfg Config) *Client {
	newClient = func(ctx context.Context, opts ...option.ClientOption) (client, error) {
		return c, nil
	}
	t, err := NewClient(context.Background(), testutil.ProjID(), cfg)
	if err != nil {
		panic(err)
	}
	return t
}

func commonChecks(t *testing.T, req *pb.ReportErrorEventRequest, fn string) {
	if req.Event.ServiceContext.Service != "myservice" {
		t.Errorf("error report didn't contain service name")
	}
	if req.Event.ServiceContext.Version != "v1.0" {
		t.Errorf("error report didn't contain version name")
	}
	if !strings.Contains(req.Event.Message, "error") {
		t.Errorf("error report didn't contain message")
	}
	if !strings.Contains(req.Event.Message, fn) {
		t.Errorf("error report didn't contain stack trace")
	}
	if got, want := req.Event.Context.User, "user"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestReport(t *testing.T) {
	fc := newFakeReportErrorsClient()
	c := newTestClient(fc, defaultConfig)
	c.Report(Entry{Error: errors.New("error"), User: "user"})
	c.Flush()
	<-fc.doneCh
	r := fc.req
	if r == nil {
		t.Fatalf("got no error report, expected one")
	}
	commonChecks(t, r, "errorreporting.TestReport")
}

func TestReportSync(t *testing.T) {
	ctx := context.Background()
	fc := newFakeReportErrorsClient()
	c := newTestClient(fc, defaultConfig)
	if err := c.ReportSync(ctx, Entry{Error: errors.New("error"), User: "user"}); err != nil {
		t.Fatalf("cannot upload errors: %v", err)
	}

	<-fc.doneCh
	r := fc.req
	if r == nil {
		t.Fatalf("got no error report, expected one")
	}
	commonChecks(t, r, "errorreporting.TestReport")
}

func TestOnError(t *testing.T) {
	fc := newFakeReportErrorsClient()
	fc.fail = true
	cfg := defaultConfig
	errc := make(chan error, 1)
	cfg.OnError = func(err error) { errc <- err }
	c := newTestClient(fc, cfg)
	c.Report(Entry{Error: errors.New("error")})
	c.Flush()
	<-fc.doneCh
	select {
	case err := <-errc:
		if err == nil {
			t.Error("got nil, want error")
		}
	case <-time.After(5 * time.Second):
		t.Error("timeout")
	}
}

func TestChopStack(t *testing.T) {
	for _, test := range []struct {
		name     string
		in       []byte
		expected string
	}{
		{
			name: "Report",
			in: []byte(` goroutine 39 [running]:
runtime/debug.Stack()
	/gopath/runtime/debug/stack.go:24 +0x79
cloud.google.com/go/errorreporting.(*Client).logInternal()
	/gopath/cloud.google.com/go/errorreporting/errors.go:259 +0x18b
cloud.google.com/go/errorreporting.(*Client).Report()
	/gopath/cloud.google.com/go/errorreporting/errors.go:248 +0x4ed
cloud.google.com/go/errorreporting.TestReport()
	/gopath/cloud.google.com/go/errorreporting/errors_test.go:137 +0x2a1
testing.tRunner()
	/gopath/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/testing/testing.go:646 +0x2ec
`),
			expected: ` goroutine 39 [running]:
cloud.google.com/go/errorreporting.TestReport()
	/gopath/cloud.google.com/go/errorreporting/errors_test.go:137 +0x2a1
testing.tRunner()
	/gopath/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/testing/testing.go:646 +0x2ec
`,
		},
	} {
		out := chopStack(test.in)
		if out != test.expected {
			t.Errorf("case %q: chopStack(%q): got %q want %q", test.name, test.in, out, test.expected)
		}
	}
}
