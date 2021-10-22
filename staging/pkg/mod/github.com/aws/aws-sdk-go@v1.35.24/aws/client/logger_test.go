package client

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
)

type mockCloser struct {
	closed bool
}

func (closer *mockCloser) Read(b []byte) (int, error) {
	return 0, io.EOF
}

func (closer *mockCloser) Close() error {
	closer.closed = true
	return nil
}

func TestTeeReaderCloser(t *testing.T) {
	expected := "FOO"
	buf := bytes.NewBuffer([]byte(expected))
	lw := bytes.NewBuffer(nil)
	c := &mockCloser{}
	closer := teeReaderCloser{
		io.TeeReader(buf, lw),
		c,
	}

	b := make([]byte, len(expected))
	_, err := closer.Read(b)
	closer.Close()

	if expected != lw.String() {
		t.Errorf("Expected %q, but received %q", expected, lw.String())
	}

	if err != nil {
		t.Errorf("Expected 'nil', but received %v", err)
	}

	if !c.closed {
		t.Error("Expected 'true', but received 'false'")
	}
}

func TestLogWriter(t *testing.T) {
	expected := "FOO"
	lw := &logWriter{nil, bytes.NewBuffer(nil)}
	lw.Write([]byte(expected))

	if expected != lw.buf.String() {
		t.Errorf("Expected %q, but received %q", expected, lw.buf.String())
	}
}

func TestLogRequest(t *testing.T) {
	cases := []struct {
		Body       io.ReadSeeker
		ExpectBody []byte
		LogLevel   aws.LogLevelType
	}{
		{
			Body:       aws.ReadSeekCloser(bytes.NewBuffer([]byte("body content"))),
			ExpectBody: []byte("body content"),
		},
		{
			Body:       aws.ReadSeekCloser(bytes.NewBuffer([]byte("body content"))),
			LogLevel:   aws.LogDebugWithHTTPBody,
			ExpectBody: []byte("body content"),
		},
		{
			Body:       bytes.NewReader([]byte("body content")),
			ExpectBody: []byte("body content"),
		},
		{
			Body:       bytes.NewReader([]byte("body content")),
			LogLevel:   aws.LogDebugWithHTTPBody,
			ExpectBody: []byte("body content"),
		},
	}

	for i, c := range cases {
		logW := bytes.NewBuffer(nil)
		req := request.New(
			aws.Config{
				Credentials: credentials.AnonymousCredentials,
				Logger:      &bufLogger{w: logW},
				LogLevel:    aws.LogLevel(c.LogLevel),
			},
			metadata.ClientInfo{
				Endpoint: "https://mock-service.mock-region.amazonaws.com",
			},
			testHandlers(),
			nil,
			&request.Operation{
				Name:       "APIName",
				HTTPMethod: "POST",
				HTTPPath:   "/",
			},
			struct{}{}, nil,
		)
		req.SetReaderBody(c.Body)
		req.Build()

		logRequest(req)

		b, err := ioutil.ReadAll(req.HTTPRequest.Body)
		if err != nil {
			t.Fatalf("%d, expect to read SDK request Body", i)
		}

		if e, a := c.ExpectBody, b; !reflect.DeepEqual(e, a) {
			t.Errorf("%d, expect %v body, got %v", i, e, a)
		}
	}
}

func TestLogResponse(t *testing.T) {
	cases := []struct {
		Body       *bytes.Buffer
		ExpectBody []byte
		ReadBody   bool
		LogLevel   aws.LogLevelType
	}{
		{
			Body:       bytes.NewBuffer([]byte("body content")),
			ExpectBody: []byte("body content"),
		},
		{
			Body:       bytes.NewBuffer([]byte("body content")),
			LogLevel:   aws.LogDebug,
			ExpectBody: []byte("body content"),
		},
		{
			Body:       bytes.NewBuffer([]byte("body content")),
			LogLevel:   aws.LogDebugWithHTTPBody,
			ReadBody:   true,
			ExpectBody: []byte("body content"),
		},
	}

	for i, c := range cases {
		var logW bytes.Buffer
		req := request.New(
			aws.Config{
				Credentials: credentials.AnonymousCredentials,
				Logger:      &bufLogger{w: &logW},
				LogLevel:    aws.LogLevel(c.LogLevel),
			},
			metadata.ClientInfo{
				Endpoint: "https://mock-service.mock-region.amazonaws.com",
			},
			testHandlers(),
			nil,
			&request.Operation{
				Name:       "APIName",
				HTTPMethod: "POST",
				HTTPPath:   "/",
			},
			struct{}{}, nil,
		)
		req.HTTPResponse = &http.Response{
			StatusCode: 200,
			Status:     "OK",
			Header: http.Header{
				"ABC": []string{"123"},
			},
			Body: ioutil.NopCloser(c.Body),
		}

		logResponse(req)
		req.Handlers.Unmarshal.Run(req)

		if c.ReadBody {
			if e, a := len(c.ExpectBody), c.Body.Len(); e != a {
				t.Errorf("%d, expect original body not to of been read", i)
			}
		}

		if logW.Len() == 0 {
			t.Errorf("%d, expect HTTP Response headers to be logged", i)
		}

		b, err := ioutil.ReadAll(req.HTTPResponse.Body)
		if err != nil {
			t.Fatalf("%d, expect to read SDK request Body", i)
		}

		if e, a := c.ExpectBody, b; !bytes.Equal(e, a) {
			t.Errorf("%d, expect %v body, got %v", i, e, a)
		}
	}
}

type bufLogger struct {
	w *bytes.Buffer
}

func (l *bufLogger) Log(args ...interface{}) {
	fmt.Fprintln(l.w, args...)
}

func testHandlers() request.Handlers {
	var handlers request.Handlers

	handlers.Build.PushBackNamed(corehandlers.SDKVersionUserAgentHandler)

	return handlers
}
