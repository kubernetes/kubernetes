package protocol_test

import (
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/ec2query"
	"github.com/aws/aws-sdk-go/private/protocol/jsonrpc"
	"github.com/aws/aws-sdk-go/private/protocol/query"
	"github.com/aws/aws-sdk-go/private/protocol/restjson"
	"github.com/aws/aws-sdk-go/private/protocol/restxml"
)

type mockCloser struct {
	*strings.Reader
	Closed bool
}

func (m *mockCloser) Close() error {
	m.Closed = true
	return nil
}

func TestUnmarshalDrainBody(t *testing.T) {
	b := &mockCloser{Reader: strings.NewReader("example body")}
	r := &request.Request{HTTPResponse: &http.Response{
		Body: b,
	}}

	protocol.UnmarshalDiscardBody(r)
	if err := r.Error; err != nil {
		t.Errorf("expect nil, %v", err)
	}
	if e, a := 0, b.Len(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if !b.Closed {
		t.Errorf("expect true")
	}
}

func TestUnmarshalDrainBodyNoBody(t *testing.T) {
	r := &request.Request{HTTPResponse: &http.Response{}}

	protocol.UnmarshalDiscardBody(r)
	if err := r.Error; err != nil {
		t.Errorf("expect nil, %v", err)
	}
}

func TestUnmarshalSeriaizationError(t *testing.T) {

	type testOutput struct {
		_ struct{}
	}

	cases := []struct {
		name          string
		r             request.Request
		unmarshalFn   func(*request.Request)
		expectedError awserr.RequestFailure
	}{
		{
			name: "jsonrpc",
			r: request.Request{
				Data: &testOutput{},
				HTTPResponse: &http.Response{
					StatusCode: 502,
					Body:       ioutil.NopCloser(strings.NewReader("invalid json")),
				},
			},
			unmarshalFn: jsonrpc.Unmarshal,
			expectedError: awserr.NewRequestFailure(
				awserr.New(request.ErrCodeSerialization, "", nil),
				502,
				"",
			),
		},
		{
			name: "ec2query",
			r: request.Request{
				Data: &testOutput{},
				HTTPResponse: &http.Response{
					StatusCode: 111,
					Body:       ioutil.NopCloser(strings.NewReader("<<>>>>>>")),
				},
			},
			unmarshalFn: ec2query.Unmarshal,
			expectedError: awserr.NewRequestFailure(
				awserr.New(request.ErrCodeSerialization, "", nil),
				111,
				"",
			),
		},
		{
			name: "query",
			r: request.Request{
				Operation: &request.Operation{
					Name: "Foo",
				},
				Data: &testOutput{},
				HTTPResponse: &http.Response{
					StatusCode: 1,
					Body:       ioutil.NopCloser(strings.NewReader("<<>>>>>>")),
				},
			},
			unmarshalFn: query.Unmarshal,
			expectedError: awserr.NewRequestFailure(
				awserr.New(request.ErrCodeSerialization, "", nil),
				1,
				"",
			),
		},
		{
			name: "restjson",
			r: request.Request{
				Data: &testOutput{},
				HTTPResponse: &http.Response{
					StatusCode: 123,
					Body:       ioutil.NopCloser(strings.NewReader("invalid json")),
				},
			},
			unmarshalFn: restjson.Unmarshal,
			expectedError: awserr.NewRequestFailure(
				awserr.New(request.ErrCodeSerialization, "", nil),
				123,
				"",
			),
		},
		{
			name: "restxml",
			r: request.Request{
				Data: &testOutput{},
				HTTPResponse: &http.Response{
					StatusCode: 456,
					Body:       ioutil.NopCloser(strings.NewReader("<<>>>>>>")),
				},
			},
			unmarshalFn: restxml.Unmarshal,
			expectedError: awserr.NewRequestFailure(
				awserr.New(request.ErrCodeSerialization, "", nil),
				456,
				"",
			),
		},
	}

	for _, c := range cases {
		c.unmarshalFn(&c.r)

		rfErr, ok := c.r.Error.(awserr.RequestFailure)
		if !ok {
			t.Errorf("%s: expected awserr.RequestFailure, but received %T", c.name, c.r.Error)
		}

		if e, a := c.expectedError.StatusCode(), rfErr.StatusCode(); e != a {
			t.Errorf("%s: expected %v, but received %v", c.name, e, a)
		}
	}
}
