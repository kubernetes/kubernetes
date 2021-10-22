// +build go1.7

package request_test

import (
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/request"
	v4 "github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/private/protocol/jsonrpc"
)

type connResetCloser struct {
	Err error
}

func (rc *connResetCloser) Read(b []byte) (int, error) {
	return 0, rc.Err
}

func (rc *connResetCloser) Close() error {
	return nil
}

func TestSerializationErrConnectionReset_accept(t *testing.T) {
	cases := map[string]struct {
		Err            error
		ExpectAttempts int
	}{
		"accept with temporary": {
			Err:            errAcceptConnectionResetStub,
			ExpectAttempts: 6,
		},
		"read not temporary": {
			Err:            errReadConnectionResetStub,
			ExpectAttempts: 1,
		},
		"write with temporary": {
			Err:            errWriteConnectionResetStub,
			ExpectAttempts: 6,
		},
		"write broken pipe with temporary": {
			Err:            errWriteBrokenPipeStub,
			ExpectAttempts: 6,
		},
		"generic connection reset": {
			Err:            errConnectionResetStub,
			ExpectAttempts: 6,
		},
		"use of closed network connection": {
			Err:            errUseOfClosedConnectionStub,
			ExpectAttempts: 6,
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			count := 0
			handlers := request.Handlers{}
			handlers.Send.PushBack(func(r *request.Request) {
				count++
				r.HTTPResponse = &http.Response{}
				r.HTTPResponse.Body = &connResetCloser{
					Err: c.Err,
				}
			})

			handlers.Sign.PushBackNamed(v4.SignRequestHandler)
			handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
			handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
			handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
			handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)
			handlers.AfterRetry.PushBackNamed(corehandlers.AfterRetryHandler)

			op := &request.Operation{
				Name:       "op",
				HTTPMethod: "POST",
				HTTPPath:   "/",
			}

			meta := metadata.ClientInfo{
				ServiceName:   "fooService",
				SigningName:   "foo",
				SigningRegion: "foo",
				Endpoint:      "localhost",
				APIVersion:    "2001-01-01",
				JSONVersion:   "1.1",
				TargetPrefix:  "Foo",
			}
			cfg := unit.Session.Config.Copy()
			cfg.MaxRetries = aws.Int(5)
			cfg.SleepDelay = func(time.Duration) {}

			req := request.New(
				*cfg,
				meta,
				handlers,
				client.DefaultRetryer{NumMaxRetries: 5},
				op,
				&struct{}{},
				&struct{}{},
			)

			osErr := c.Err
			req.ApplyOptions(request.WithResponseReadTimeout(time.Second))
			err := req.Send()
			if err == nil {
				t.Error("Expected error 'SerializationError', but received nil")
			}
			if aerr, ok := err.(awserr.Error); ok && aerr.Code() != request.ErrCodeSerialization {
				t.Errorf("Expected 'SerializationError', but received %q", aerr.Code())
			} else if !ok {
				t.Errorf("Expected 'awserr.Error', but received %v", reflect.TypeOf(err))
			} else if aerr.OrigErr().Error() != osErr.Error() {
				t.Errorf("Expected %q, but received %q", osErr.Error(), aerr.OrigErr().Error())
			}

			if e, a := c.ExpectAttempts, count; e != a {
				t.Errorf("Expected %v, but received %v", e, a)
			}
		})
	}
}
