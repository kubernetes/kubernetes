// +build go1.8

package route53

import (
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

func TestUnmarshalInvalidChangeBatch(t *testing.T) {
	const errorMessage = `
Tried to create resource record set duplicate.example.com. type A,
but it already exists
`

	type batchError struct {
		Code, Message string
	}

	cases := map[string]struct {
		Request                  *request.Request
		Code, Message, RequestID string
		StatusCode               int
		BatchErrors              []batchError
	}{
		"standard error": {
			Request: &request.Request{
				HTTPResponse: &http.Response{
					StatusCode: 400,
					Header:     http.Header{},
					Body: ioutil.NopCloser(strings.NewReader(
						`<?xml version="1.0" encoding="UTF-8"?>
<ErrorResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">
  <Error>
	<Code>InvalidDomainName</Code>
	<Message>The domain name is invalid</Message>
  </Error>
  <RequestId>12345</RequestId>
</ErrorResponse>`)),
				},
			},
			Code: "InvalidDomainName", Message: "The domain name is invalid",
			StatusCode: 400, RequestID: "12345",
		},
		"batched error": {
			Request: &request.Request{
				HTTPResponse: &http.Response{
					StatusCode: 400,
					Header:     http.Header{},
					Body: ioutil.NopCloser(strings.NewReader(
						`<?xml version="1.0" encoding="UTF-8"?>
<InvalidChangeBatch xmlns="https://route53.amazonaws.com/doc/2013-04-01/">
  <Messages>
	<Message>` + errorMessage + `</Message>
  </Messages>
  <RequestId>12345</RequestId>
</InvalidChangeBatch>`)),
				},
			},
			Code: "InvalidChangeBatch", Message: "ChangeBatch errors occurred",
			StatusCode: 400, RequestID: "12345",
			BatchErrors: []batchError{
				{Code: "InvalidChangeBatch", Message: errorMessage},
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			unmarshalChangeResourceRecordSetsError(c.Request)
			err := c.Request.Error
			if err == nil {
				t.Error("expected error, but received none")
			}

			reqErr := err.(awserr.RequestFailure)
			if e, a := c.StatusCode, reqErr.StatusCode(); e != a {
				t.Errorf("expected %d status, got %d", e, a)
			}
			if e, a := c.Code, reqErr.Code(); e != a {
				t.Errorf("expected %v code, got %v", e, a)
			}
			if e, a := c.Message, reqErr.Message(); e != a {
				t.Errorf("expected %q message, got %q", e, a)
			}
			if e, a := c.RequestID, reqErr.RequestID(); e != a {
				t.Errorf("expected %v request ID, got %v", e, a)
			}

			batchErr := err.(awserr.BatchedErrors)
			batchedErrs := batchErr.OrigErrs()

			if e, a := len(c.BatchErrors), len(batchedErrs); e != a {
				t.Fatalf("expect %v batch errors, got %v", e, a)
			}

			for i, ee := range c.BatchErrors {
				bErr := batchedErrs[i].(awserr.Error)
				if e, a := ee.Code, bErr.Code(); e != a {
					t.Errorf("expect %v code, got %v", e, a)
				}
				if e, a := ee.Message, bErr.Message(); e != a {
					t.Errorf("expect %v message, got %v", e, a)
				}
			}
		})
	}
}
