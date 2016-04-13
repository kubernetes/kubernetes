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

package prometheus

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/prometheus/common/model"
	"golang.org/x/net/context"
)

func TestConfig(t *testing.T) {
	c := Config{}
	if c.transport() != DefaultTransport {
		t.Fatalf("expected default transport for nil Transport field")
	}
}

func TestClientURL(t *testing.T) {
	tests := []struct {
		address  string
		endpoint string
		args     map[string]string
		expected string
	}{
		{
			address:  "http://localhost:9090",
			endpoint: "/test",
			expected: "http://localhost:9090/test",
		},
		{
			address:  "http://localhost",
			endpoint: "/test",
			expected: "http://localhost/test",
		},
		{
			address:  "http://localhost:9090",
			endpoint: "test",
			expected: "http://localhost:9090/test",
		},
		{
			address:  "http://localhost:9090/prefix",
			endpoint: "/test",
			expected: "http://localhost:9090/prefix/test",
		},
		{
			address:  "https://localhost:9090/",
			endpoint: "/test/",
			expected: "https://localhost:9090/test",
		},
		{
			address:  "http://localhost:9090",
			endpoint: "/test/:param",
			args: map[string]string{
				"param": "content",
			},
			expected: "http://localhost:9090/test/content",
		},
		{
			address:  "http://localhost:9090",
			endpoint: "/test/:param/more/:param",
			args: map[string]string{
				"param": "content",
			},
			expected: "http://localhost:9090/test/content/more/content",
		},
		{
			address:  "http://localhost:9090",
			endpoint: "/test/:param/more/:foo",
			args: map[string]string{
				"param": "content",
				"foo":   "bar",
			},
			expected: "http://localhost:9090/test/content/more/bar",
		},
		{
			address:  "http://localhost:9090",
			endpoint: "/test/:param",
			args: map[string]string{
				"nonexistant": "content",
			},
			expected: "http://localhost:9090/test/:param",
		},
	}

	for _, test := range tests {
		ep, err := url.Parse(test.address)
		if err != nil {
			t.Fatal(err)
		}

		hclient := &httpClient{
			endpoint:  ep,
			transport: DefaultTransport,
		}

		u := hclient.url(test.endpoint, test.args)
		if u.String() != test.expected {
			t.Errorf("unexpected result: got %s, want %s", u, test.expected)
			continue
		}

		// The apiClient must return exactly the same result as the httpClient.
		aclient := &apiClient{hclient}

		u = aclient.url(test.endpoint, test.args)
		if u.String() != test.expected {
			t.Errorf("unexpected result: got %s, want %s", u, test.expected)
		}
	}
}

type testClient struct {
	*testing.T

	ch  chan apiClientTest
	req *http.Request
}

type apiClientTest struct {
	code     int
	response interface{}
	expected string
	err      *Error
}

func (c *testClient) url(ep string, args map[string]string) *url.URL {
	return nil
}

func (c *testClient) do(ctx context.Context, req *http.Request) (*http.Response, []byte, error) {
	if ctx == nil {
		c.Fatalf("context was not passed down")
	}
	if req != c.req {
		c.Fatalf("request was not passed down")
	}

	test := <-c.ch

	var b []byte
	var err error

	switch v := test.response.(type) {
	case string:
		b = []byte(v)
	default:
		b, err = json.Marshal(v)
		if err != nil {
			c.Fatal(err)
		}
	}

	resp := &http.Response{
		StatusCode: test.code,
	}

	return resp, b, nil
}

func TestAPIClientDo(t *testing.T) {
	tests := []apiClientTest{
		{
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`null`),
				ErrorType: ErrBadData,
				Error:     "failed",
			},
			err: &Error{
				Type: ErrBadData,
				Msg:  "failed",
			},
			code:     statusAPIError,
			expected: `null`,
		},
		{
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`"test"`),
				ErrorType: ErrTimeout,
				Error:     "timed out",
			},
			err: &Error{
				Type: ErrTimeout,
				Msg:  "timed out",
			},
			code:     statusAPIError,
			expected: `test`,
		},
		{
			response: "bad json",
			err: &Error{
				Type: ErrBadResponse,
				Msg:  "bad response code 400",
			},
			code: http.StatusBadRequest,
		},
		{
			response: "bad json",
			err: &Error{
				Type: ErrBadResponse,
				Msg:  "invalid character 'b' looking for beginning of value",
			},
			code: statusAPIError,
		},
		{
			response: &apiResponse{
				Status: "success",
				Data:   json.RawMessage(`"test"`),
			},
			err: &Error{
				Type: ErrBadResponse,
				Msg:  "inconsistent body for response code",
			},
			code: statusAPIError,
		},
		{
			response: &apiResponse{
				Status:    "success",
				Data:      json.RawMessage(`"test"`),
				ErrorType: ErrTimeout,
				Error:     "timed out",
			},
			err: &Error{
				Type: ErrBadResponse,
				Msg:  "inconsistent body for response code",
			},
			code: statusAPIError,
		},
		{
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`"test"`),
				ErrorType: ErrTimeout,
				Error:     "timed out",
			},
			err: &Error{
				Type: ErrBadResponse,
				Msg:  "inconsistent body for response code",
			},
			code: http.StatusOK,
		},
	}

	tc := &testClient{
		T:   t,
		ch:  make(chan apiClientTest, 1),
		req: &http.Request{},
	}
	client := &apiClient{tc}

	for _, test := range tests {

		tc.ch <- test

		_, body, err := client.do(context.Background(), tc.req)

		if test.err != nil {
			if err == nil {
				t.Errorf("expected error %q but got none", test.err)
				continue
			}
			if test.err.Error() != err.Error() {
				t.Errorf("unexpected error: want %q, got %q", test.err, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpeceted error %s", err)
			continue
		}

		want, got := test.expected, string(body)
		if want != got {
			t.Errorf("unexpected body: want %q, got %q", want, got)
		}
	}
}

type apiTestClient struct {
	*testing.T
	curTest apiTest
}

type apiTest struct {
	do    func() (interface{}, error)
	inErr error
	inRes interface{}

	reqPath   string
	reqParam  url.Values
	reqMethod string
	res       interface{}
	err       error
}

func (c *apiTestClient) url(ep string, args map[string]string) *url.URL {
	u := &url.URL{
		Host: "test:9090",
		Path: apiPrefix + ep,
	}
	return u
}

func (c *apiTestClient) do(ctx context.Context, req *http.Request) (*http.Response, []byte, error) {

	test := c.curTest

	if req.URL.Path != test.reqPath {
		c.Errorf("unexpected request path: want %s, got %s", test.reqPath, req.URL.Path)
	}
	if req.Method != test.reqMethod {
		c.Errorf("unexpected request method: want %s, got %s", test.reqMethod, req.Method)
	}

	b, err := json.Marshal(test.inRes)
	if err != nil {
		c.Fatal(err)
	}

	resp := &http.Response{}
	if test.inErr != nil {
		resp.StatusCode = statusAPIError
	} else {
		resp.StatusCode = http.StatusOK
	}

	return resp, b, test.inErr
}

func TestAPIs(t *testing.T) {

	testTime := time.Now()

	client := &apiTestClient{T: t}

	queryApi := &httpQueryAPI{
		client: client,
	}

	doQuery := func(q string, ts time.Time) func() (interface{}, error) {
		return func() (interface{}, error) {
			return queryApi.Query(context.Background(), q, ts)
		}
	}

	doQueryRange := func(q string, rng Range) func() (interface{}, error) {
		return func() (interface{}, error) {
			return queryApi.QueryRange(context.Background(), q, rng)
		}
	}

	queryTests := []apiTest{
		{
			do: doQuery("2", testTime),
			inRes: &queryResult{
				Type: model.ValScalar,
				Result: &model.Scalar{
					Value:     2,
					Timestamp: model.TimeFromUnix(testTime.Unix()),
				},
			},

			reqMethod: "GET",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			res: &model.Scalar{
				Value:     2,
				Timestamp: model.TimeFromUnix(testTime.Unix()),
			},
		},
		{
			do:    doQuery("2", testTime),
			inErr: fmt.Errorf("some error"),

			reqMethod: "GET",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			err: fmt.Errorf("some error"),
		},

		{
			do: doQueryRange("2", Range{
				Start: testTime.Add(-time.Minute),
				End:   testTime,
				Step:  time.Minute,
			}),
			inErr: fmt.Errorf("some error"),

			reqMethod: "GET",
			reqPath:   "/api/v1/query_range",
			reqParam: url.Values{
				"query": []string{"2"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
				"step":  []string{time.Minute.String()},
			},
			err: fmt.Errorf("some error"),
		},
	}

	var tests []apiTest
	tests = append(tests, queryTests...)

	for _, test := range tests {
		client.curTest = test

		res, err := test.do()

		if test.err != nil {
			if err == nil {
				t.Errorf("expected error %q but got none", test.err)
				continue
			}
			if err.Error() != test.err.Error() {
				t.Errorf("unexpected error: want %s, got %s", test.err, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %s", err)
			continue
		}

		if !reflect.DeepEqual(res, test.res) {
			t.Errorf("unexpected result: want %v, got %v", test.res, res)
		}
	}
}
