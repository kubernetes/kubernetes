// +build go1.7

package protocol

import (
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

func TestRequireHTTPMinProtocol(t *testing.T) {
	cases := map[string]struct {
		Major, Minor int
		Response     *http.Response
		Err          string
	}{
		"HTTP/2.0": {
			Major: 2,
			Response: &http.Response{
				StatusCode: 200,
				Proto:      "HTTP/2.0",
				ProtoMajor: 2, ProtoMinor: 0,
			},
		},
		"HTTP/1.1": {
			Major: 2,
			Response: &http.Response{
				StatusCode: 200,
				Proto:      "HTTP/1.1",
				ProtoMajor: 1, ProtoMinor: 1,
			},
			Err: "operation requires minimum HTTP protocol",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			req := &request.Request{
				HTTPResponse: c.Response,
			}
			RequireHTTPMinProtocol{Major: c.Major, Minor: c.Minor}.Handler(req)

			if len(c.Err) != 0 {
				if req.Error == nil {
					t.Fatalf("expect error")
				}
				if e, a := c.Err, req.Error.Error(); !strings.Contains(a, e) {
					t.Errorf("expect %q error, got %q", e, a)
				}
				aerr, ok := req.Error.(awserr.RequestFailure)
				if !ok {
					t.Fatalf("expect RequestFailure, got %T", req.Error)
				}

				if e, a := ErrCodeMinimumHTTPProtocolError, aerr.Code(); e != a {
					t.Errorf("expect %v code, got %v", e, a)
				}
				if e, a := c.Response.StatusCode, aerr.StatusCode(); e != a {
					t.Errorf("expect %v status code, got %v", e, a)
				}

			} else {
				if err := req.Error; err != nil {
					t.Fatalf("expect no failure, got %v", err)
				}
			}
		})
	}
}
