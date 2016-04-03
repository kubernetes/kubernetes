package simpledb_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/simpledb"
)

var statusCodeErrorTests = []struct {
	scode   int
	status  string
	code    string
	message string
}{
	{301, "Moved Permanently", "MovedPermanently", "Moved Permanently"},
	{403, "Forbidden", "Forbidden", "Forbidden"},
	{400, "Bad Request", "BadRequest", "Bad Request"},
	{404, "Not Found", "NotFound", "Not Found"},
	{500, "Internal Error", "InternalError", "Internal Error"},
}

func TestStatusCodeError(t *testing.T) {
	for _, test := range statusCodeErrorTests {
		s := simpledb.New(unit.Session)
		s.Handlers.Send.Clear()
		s.Handlers.Send.PushBack(func(r *request.Request) {
			body := ioutil.NopCloser(bytes.NewReader([]byte{}))
			r.HTTPResponse = &http.Response{
				ContentLength: 0,
				StatusCode:    test.scode,
				Status:        test.status,
				Body:          body,
			}
		})
		_, err := s.CreateDomain(&simpledb.CreateDomainInput{
			DomainName: aws.String("test-domain"),
		})

		assert.Error(t, err)
		assert.Equal(t, test.code, err.(awserr.Error).Code())
		assert.Equal(t, test.message, err.(awserr.Error).Message())
	}
}

var responseErrorTests = []struct {
	scode     int
	status    string
	code      string
	message   string
	requestID string
	errors    []struct {
		code    string
		message string
	}
}{
	{
		scode:     400,
		status:    "Bad Request",
		code:      "MissingError",
		message:   "missing error code in SimpleDB XML error response",
		requestID: "101",
		errors:    []struct{ code, message string }{},
	},
	{
		scode:     403,
		status:    "Forbidden",
		code:      "AuthFailure",
		message:   "AWS was not able to validate the provided access keys.",
		requestID: "1111",
		errors: []struct{ code, message string }{
			{"AuthFailure", "AWS was not able to validate the provided access keys."},
		},
	},
	{
		scode:     500,
		status:    "Internal Error",
		code:      "MissingParameter",
		message:   "Message #1",
		requestID: "8756",
		errors: []struct{ code, message string }{
			{"MissingParameter", "Message #1"},
			{"InternalError", "Message #2"},
		},
	},
}

func TestResponseError(t *testing.T) {
	for _, test := range responseErrorTests {
		s := simpledb.New(unit.Session)
		s.Handlers.Send.Clear()
		s.Handlers.Send.PushBack(func(r *request.Request) {
			xml := createXMLResponse(test.requestID, test.errors)
			body := ioutil.NopCloser(bytes.NewReader([]byte(xml)))
			r.HTTPResponse = &http.Response{
				ContentLength: int64(len(xml)),
				StatusCode:    test.scode,
				Status:        test.status,
				Body:          body,
			}
		})
		_, err := s.CreateDomain(&simpledb.CreateDomainInput{
			DomainName: aws.String("test-domain"),
		})

		assert.Error(t, err)
		assert.Equal(t, test.code, err.(awserr.Error).Code())
		assert.Equal(t, test.message, err.(awserr.Error).Message())
		if len(test.errors) > 0 {
			assert.Equal(t, test.requestID, err.(awserr.RequestFailure).RequestID())
			assert.Equal(t, test.scode, err.(awserr.RequestFailure).StatusCode())
		}
	}
}

// createXMLResponse constructs an XML string that has one or more error messages in it.
func createXMLResponse(requestID string, errors []struct{ code, message string }) []byte {
	var buf bytes.Buffer
	buf.WriteString(`<?xml version="1.0"?><Response><Errors>`)
	for _, e := range errors {
		buf.WriteString(`<Error><Code>`)
		buf.WriteString(e.code)
		buf.WriteString(`</Code><Message>`)
		buf.WriteString(e.message)
		buf.WriteString(`</Message></Error>`)
	}
	buf.WriteString(`</Errors><RequestID>`)
	buf.WriteString(requestID)
	buf.WriteString(`</RequestID></Response>`)
	return buf.Bytes()
}
