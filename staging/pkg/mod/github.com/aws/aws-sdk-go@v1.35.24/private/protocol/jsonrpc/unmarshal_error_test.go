// +build go1.7

package jsonrpc

import (
	"bytes"
	"encoding/hex"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol"
)

const unknownErrJSON = `{"__type":"UnknownError", "message":"error message", "something":123}`
const simpleErrJSON = `{"__type":"SimpleError", "message":"some message", "foo":123}`
const simpleCasedErrJSON = `{"__type":"SimpleError", "Message":"some message", "foo":123}`

type SimpleError struct {
	_ struct{} `type:"structure"`
	error

	Message2 *string `type:"string" locationName:"message"`
	Foo      *int64  `type:"integer" locationName:"foo"`
}

const otherErrJSON = `{"__type":"OtherError", "message":"some message"}`
const complexCodeErrJSON = `{"__type":"foo.bar#OtherError", "message":"some message"}`

type OtherError struct {
	_ struct{} `type:"structure"`
	error

	Message2 *string `type:"string" locationName:"message"`
}

const complexErrJSON = `{"__type":"ComplexError", "message":"some message", "foo": {"bar":"abc123", "baz":123}}`

type ComplexError struct {
	_ struct{} `type:"structure"`
	error

	Message2 *string      `type:"string" locationName:"message"`
	Foo      *ErrorNested `type:"structure" locationName:"foo"`
}
type ErrorNested struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string" locationName:"bar"`
	Baz *int64  `type:"integer" locationName:"baz"`
}

func TestUnmarshalTypedError(t *testing.T) {

	respMeta := protocol.ResponseMetadata{
		StatusCode: 400,
		RequestID:  "abc123",
	}

	exceptions := map[string]func(protocol.ResponseMetadata) error{
		"SimpleError": func(meta protocol.ResponseMetadata) error {
			return &SimpleError{}
		},
		"OtherError": func(meta protocol.ResponseMetadata) error {
			return &OtherError{}
		},
		"ComplexError": func(meta protocol.ResponseMetadata) error {
			return &ComplexError{}
		},
	}

	cases := map[string]struct {
		Response *http.Response
		Expect   error
		Err      string
	}{
		"simple error": {
			Response: &http.Response{
				Header: http.Header{},
				Body:   ioutil.NopCloser(strings.NewReader(simpleErrJSON)),
			},
			Expect: &SimpleError{
				Message2: aws.String("some message"),
				Foo:      aws.Int64(123),
			},
		},
		"other error": {
			Response: &http.Response{
				Header: http.Header{},
				Body:   ioutil.NopCloser(strings.NewReader(otherErrJSON)),
			},
			Expect: &OtherError{
				Message2: aws.String("some message"),
			},
		},
		"other complex Code error": {
			Response: &http.Response{
				Header: http.Header{},
				Body:   ioutil.NopCloser(strings.NewReader(complexCodeErrJSON)),
			},
			Expect: &OtherError{
				Message2: aws.String("some message"),
			},
		},
		"complex error": {
			Response: &http.Response{
				Header: http.Header{},
				Body:   ioutil.NopCloser(strings.NewReader(complexErrJSON)),
			},
			Expect: &ComplexError{
				Message2: aws.String("some message"),
				Foo: &ErrorNested{
					Bar: aws.String("abc123"),
					Baz: aws.Int64(123),
				},
			},
		},
		"unknown error": {
			Response: &http.Response{
				Header: http.Header{},
				Body:   ioutil.NopCloser(strings.NewReader(unknownErrJSON)),
			},
			Expect: awserr.NewRequestFailure(
				awserr.New("UnknownError", "error message", nil),
				respMeta.StatusCode,
				respMeta.RequestID,
			),
		},
		"invalid error": {
			Response: &http.Response{
				StatusCode: 400,
				Header:     http.Header{},
				Body:       ioutil.NopCloser(strings.NewReader(`{`)),
			},
			Err: "failed decoding",
		},
		"mixed case fields": {
			Response: &http.Response{
				Header: http.Header{},
				Body:   ioutil.NopCloser(strings.NewReader(simpleCasedErrJSON)),
			},
			Expect: &SimpleError{
				Message2: aws.String("some message"),
				Foo:      aws.Int64(123),
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			u := NewUnmarshalTypedError(exceptions)
			v, err := u.UnmarshalError(c.Response, respMeta)

			if len(c.Err) != 0 {
				if err == nil {
					t.Fatalf("expect error, got none")
				}
				if e, a := c.Err, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect %v in error, got %v", e, a)
				}
			} else if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}

			if e, a := c.Expect, v; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %+#v, got %#+v", e, a)
			}
		})
	}
}

func TestUnmarshalError_SerializationError(t *testing.T) {
	cases := map[string]struct {
		Request     *request.Request
		ExpectMsg   string
		ExpectBytes []byte
	}{
		"empty body": {
			Request: &request.Request{
				Data: &struct{}{},
				HTTPResponse: &http.Response{
					StatusCode: 400,
					Header: http.Header{
						"X-Amzn-Requestid": []string{"abc123"},
					},
					Body: ioutil.NopCloser(
						bytes.NewReader([]byte{}),
					),
				},
			},
			ExpectMsg: "error message missing",
		},
		"HTML body": {
			Request: &request.Request{
				Data: &struct{}{},
				HTTPResponse: &http.Response{
					StatusCode: 400,
					Header: http.Header{
						"X-Amzn-Requestid": []string{"abc123"},
					},
					Body: ioutil.NopCloser(
						bytes.NewReader([]byte(`<html></html>`)),
					),
				},
			},
			ExpectBytes: []byte(`<html></html>`),
			ExpectMsg:   "failed decoding",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			req := c.Request

			UnmarshalError(req)
			if req.Error == nil {
				t.Fatal("expect error, got none")
			}

			aerr := req.Error.(awserr.RequestFailure)
			if e, a := request.ErrCodeSerialization, aerr.Code(); e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			uerr := aerr.OrigErr().(awserr.UnmarshalError)
			if e, a := c.ExpectMsg, uerr.Message(); !strings.Contains(a, e) {
				t.Errorf("Expect %q, in %q", e, a)
			}
			if e, a := c.ExpectBytes, uerr.Bytes(); !bytes.Equal(e, a) {
				t.Errorf("expect:\n%v\nactual:\n%v", hex.Dump(e), hex.Dump(a))
			}
		})
	}
}
