// Copyright 2016 Google Inc. All Rights Reserved.
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

package errors_test

import (
	"bytes"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"testing"

	"cloud.google.com/go/errors"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
)

const testProjectID = "testproject"

type fakeRoundTripper struct {
	req  *http.Request
	fail bool
	body string
}

func newFakeRoundTripper() *fakeRoundTripper {
	return &fakeRoundTripper{}
}

func (rt *fakeRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	rt.req = r
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		panic(err)
	}
	rt.body = string(body)
	if rt.fail {
		return &http.Response{
			Status:     "503 Service Unavailable",
			StatusCode: 503,
			Body:       ioutil.NopCloser(strings.NewReader("{}")),
		}, nil
	}
	return &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Body:       ioutil.NopCloser(strings.NewReader("{}")),
	}, nil
}

func newTestClient(rt http.RoundTripper) *errors.Client {
	t, err := errors.NewClient(context.Background(), testProjectID, "myservice", "v1.000", option.WithHTTPClient(&http.Client{Transport: rt}))
	if err != nil {
		panic(err)
	}
	t.RepanicDefault = false
	return t
}

var ctx context.Context

func init() {
	ctx = context.Background()
}

func TestCatchNothing(t *testing.T) {
	rt := &fakeRoundTripper{}
	c := newTestClient(rt)
	defer func() {
		r := rt.req
		if r != nil {
			t.Errorf("got error report, expected none")
		}
	}()
	defer c.Catch(ctx)
}

func commonChecks(t *testing.T, body, panickingFunction string) {
	if !strings.Contains(body, "myservice") {
		t.Errorf("error report didn't contain service name")
	}
	if !strings.Contains(body, "v1.000") {
		t.Errorf("error report didn't contain version name")
	}
	if !strings.Contains(body, "hello, error") {
		t.Errorf("error report didn't contain message")
	}
	if !strings.Contains(body, panickingFunction) {
		t.Errorf("error report didn't contain stack trace")
	}
}

func TestCatchPanic(t *testing.T) {
	rt := &fakeRoundTripper{}
	c := newTestClient(rt)
	defer func() {
		r := rt.req
		if r == nil {
			t.Fatalf("got no error report, expected one")
		}
		commonChecks(t, rt.body, "errors_test.TestCatchPanic")
		if !strings.Contains(rt.body, "divide by zero") {
			t.Errorf("error report didn't contain recovered value")
		}
	}()
	defer c.Catch(ctx, errors.WithMessage("hello, error"))
	var x int
	x = x / x
}

func TestCatchPanicNilClient(t *testing.T) {
	buf := new(bytes.Buffer)
	log.SetOutput(buf)
	defer func() {
		recover()
		body := buf.Bytes()
		if !strings.Contains(string(body), "divide by zero") {
			t.Errorf("error report didn't contain recovered value")
		}
		if !strings.Contains(string(body), "hello, error") {
			t.Errorf("error report didn't contain message")
		}
		if !strings.Contains(string(body), "errors_test.TestCatchPanicNilClient") {
			t.Errorf("error report didn't contain recovered value")
		}
	}()
	var c *errors.Client
	defer c.Catch(ctx, errors.WithMessage("hello, error"))
	var x int
	x = x / x
}

func TestLogFailedReports(t *testing.T) {
	rt := &fakeRoundTripper{}
	c := newTestClient(rt)
	rt.fail = true
	buf := new(bytes.Buffer)
	log.SetOutput(buf)
	defer func() {
		recover()
		body := buf.Bytes()
		commonChecks(t, string(body), "errors_test.TestLogFailedReports")
		if !strings.Contains(string(body), "divide by zero") {
			t.Errorf("error report didn't contain recovered value")
		}
	}()
	defer c.Catch(ctx, errors.WithMessage("hello, error"))
	var x int
	x = x / x
}

func TestCatchNilPanic(t *testing.T) {
	rt := &fakeRoundTripper{}
	c := newTestClient(rt)
	defer func() {
		r := rt.req
		if r == nil {
			t.Fatalf("got no error report, expected one")
		}
		commonChecks(t, rt.body, "errors_test.TestCatchNilPanic")
		if !strings.Contains(rt.body, "nil") {
			t.Errorf("error report didn't contain recovered value")
		}
	}()
	b := true
	defer c.Catch(ctx, errors.WithMessage("hello, error"), errors.PanicFlag(&b))
	panic(nil)
}

func TestNotCatchNilPanic(t *testing.T) {
	rt := &fakeRoundTripper{}
	c := newTestClient(rt)
	defer func() {
		r := rt.req
		if r != nil {
			t.Errorf("got error report, expected none")
		}
	}()
	defer c.Catch(ctx, errors.WithMessage("hello, error"))
	panic(nil)
}

func TestReport(t *testing.T) {
	rt := &fakeRoundTripper{}
	c := newTestClient(rt)
	c.Report(ctx, nil, "hello, ", "error")
	r := rt.req
	if r == nil {
		t.Fatalf("got no error report, expected one")
	}
	commonChecks(t, rt.body, "errors_test.TestReport")
}

func TestReportf(t *testing.T) {
	rt := &fakeRoundTripper{}
	c := newTestClient(rt)
	c.Reportf(ctx, nil, "hello, error 2+%d=%d", 2, 2+2)
	r := rt.req
	if r == nil {
		t.Fatalf("got no error report, expected one")
	}
	commonChecks(t, rt.body, "errors_test.TestReportf")
	if !strings.Contains(rt.body, "2+2=4") {
		t.Errorf("error report didn't contain formatted message")
	}
}
