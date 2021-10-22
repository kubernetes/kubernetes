// +build go1.7

package csm_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/csm"
	"github.com/aws/aws-sdk-go/aws/request"
	v4 "github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/private/protocol/jsonrpc"
)

func TestReportingMetrics(t *testing.T) {
	sess := unit.Session.Copy(&aws.Config{
		SleepDelay: func(time.Duration) {},
	})
	sess.Handlers.Validate.Clear()
	sess.Handlers.Sign.Clear()
	sess.Handlers.Send.Clear()

	reporter := csm.Get()
	if reporter == nil {
		t.Errorf("expected non-nil reporter")
	}
	reporter.InjectHandlers(&sess.Handlers)

	cases := map[string]struct {
		Request       *request.Request
		ExpectMetrics []map[string]interface{}
	}{
		"successful request": {
			Request: func() *request.Request {
				md := metadata.ClientInfo{}
				op := &request.Operation{Name: "OperationName"}
				req := request.New(*sess.Config, md, sess.Handlers, client.DefaultRetryer{NumMaxRetries: 3}, op, nil, nil)
				req.Handlers.Send.PushBack(func(r *request.Request) {
					req.HTTPResponse = &http.Response{
						StatusCode: 200,
						Header:     http.Header{},
					}
				})
				return req
			}(),
			ExpectMetrics: []map[string]interface{}{
				{
					"Type":           "ApiCallAttempt",
					"HttpStatusCode": float64(200),
				},
				{
					"Type":                "ApiCall",
					"FinalHttpStatusCode": float64(200),
				},
			},
		},
		"failed request, no retry": {
			Request: func() *request.Request {
				md := metadata.ClientInfo{}
				op := &request.Operation{Name: "OperationName"}
				req := request.New(*sess.Config, md, sess.Handlers, client.DefaultRetryer{NumMaxRetries: 3}, op, nil, nil)
				req.Handlers.Send.PushBack(func(r *request.Request) {
					req.HTTPResponse = &http.Response{
						StatusCode: 400,
						Header:     http.Header{},
					}
					req.Retryable = aws.Bool(false)
					req.Error = awserr.New("Error", "Message", nil)
				})

				return req
			}(),
			ExpectMetrics: []map[string]interface{}{
				{
					"Type":                "ApiCallAttempt",
					"HttpStatusCode":      float64(400),
					"AwsException":        "Error",
					"AwsExceptionMessage": "Error: Message",
				},
				{
					"Type":                     "ApiCall",
					"FinalHttpStatusCode":      float64(400),
					"FinalAwsException":        "Error",
					"FinalAwsExceptionMessage": "Error: Message",
					"AttemptCount":             float64(1),
				},
			},
		},
		"failed request, with retry": {
			Request: func() *request.Request {
				md := metadata.ClientInfo{}
				op := &request.Operation{Name: "OperationName"}
				req := request.New(*sess.Config, md, sess.Handlers, client.DefaultRetryer{NumMaxRetries: 1}, op, nil, nil)
				resps := []*http.Response{
					{
						StatusCode: 500,
						Header:     http.Header{},
					},
					{
						StatusCode: 500,
						Header:     http.Header{},
					},
				}
				req.Handlers.Send.PushBack(func(r *request.Request) {
					req.HTTPResponse = resps[0]
					resps = resps[1:]
				})

				return req
			}(),
			ExpectMetrics: []map[string]interface{}{
				{
					"Type":                "ApiCallAttempt",
					"HttpStatusCode":      float64(500),
					"AwsException":        "UnknownError",
					"AwsExceptionMessage": "UnknownError: unknown error",
				},
				{
					"Type":                "ApiCallAttempt",
					"HttpStatusCode":      float64(500),
					"AwsException":        "UnknownError",
					"AwsExceptionMessage": "UnknownError: unknown error",
				},
				{
					"Type":                     "ApiCall",
					"FinalHttpStatusCode":      float64(500),
					"FinalAwsException":        "UnknownError",
					"FinalAwsExceptionMessage": "UnknownError: unknown error",
					"AttemptCount":             float64(2),
				},
			},
		},
		"success request, with retry": {
			Request: func() *request.Request {
				md := metadata.ClientInfo{}
				op := &request.Operation{Name: "OperationName"}
				req := request.New(*sess.Config, md, sess.Handlers, client.DefaultRetryer{NumMaxRetries: 3}, op, nil, nil)
				errs := []error{
					awserr.New("AWSError", "aws error", nil),
					awserr.New(request.ErrCodeRequestError, "sdk error", nil),
					nil,
				}
				resps := []*http.Response{
					{
						StatusCode: 500,
						Header:     http.Header{},
					},
					{
						StatusCode: 500,
						Header:     http.Header{},
					},
					{
						StatusCode: 200,
						Header:     http.Header{},
					},
				}
				req.Handlers.Send.PushBack(func(r *request.Request) {
					req.HTTPResponse = resps[0]
					resps = resps[1:]
					req.Error = errs[0]
					errs = errs[1:]
				})

				return req
			}(),
			ExpectMetrics: []map[string]interface{}{
				{
					"Type":                "ApiCallAttempt",
					"AwsException":        "AWSError",
					"AwsExceptionMessage": "AWSError: aws error",
					"HttpStatusCode":      float64(500),
				},
				{
					"Type":                "ApiCallAttempt",
					"SdkException":        request.ErrCodeRequestError,
					"SdkExceptionMessage": request.ErrCodeRequestError + ": sdk error",
					"HttpStatusCode":      float64(500),
				},
				{
					"Type":                "ApiCallAttempt",
					"AwsException":        nil,
					"AwsExceptionMessage": nil,
					"SdkException":        nil,
					"SdkExceptionMessage": nil,
					"HttpStatusCode":      float64(200),
				},
				{
					"Type":                     "ApiCall",
					"FinalHttpStatusCode":      float64(200),
					"FinalAwsException":        nil,
					"FinalAwsExceptionMessage": nil,
					"FinalSdkException":        nil,
					"FinalSdkExceptionMessage": nil,
					"AttemptCount":             float64(3),
				},
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, cancelFn := context.WithTimeout(context.Background(), time.Second)
			defer cancelFn()

			c.Request.Send()
			for i := 0; i < len(c.ExpectMetrics); i++ {
				select {
				case m := <-csm.MetricsCh:
					for ek, ev := range c.ExpectMetrics[i] {
						if ev == nil {
							// must not be set
							if _, ok := m[ek]; ok {
								t.Errorf("%d, expect %v metric member, not to be set, %v", i, ek, m[ek])
							}
							continue
						}
						if _, ok := m[ek]; !ok {
							t.Errorf("%d, expect %v metric member, keys: %v", i, ek, keys(m))
						}
						if e, a := ev, m[ek]; e != a {
							t.Errorf("%d, expect %v:%v(%T), metric value, got %v(%T)", i, ek, e, e, a, a)
						}
					}
				case <-ctx.Done():
					t.Errorf("timeout waiting for metrics")
					return
				}
			}

			var extraMetrics []map[string]interface{}
		Loop:
			for {
				select {
				case m := <-csm.MetricsCh:
					extraMetrics = append(extraMetrics, m)
				default:
					break Loop
				}
			}
			if len(extraMetrics) != 0 {
				t.Fatalf("unexpected metrics, %#v", extraMetrics)
			}
		})
	}
}

type mockService struct {
	*client.Client
}

type input struct{}
type output struct{}

func (s *mockService) Request(i input) *request.Request {
	op := &request.Operation{
		Name:       "foo",
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	o := output{}
	req := s.NewRequest(op, &i, &o)
	return req
}

func BenchmarkWithCSM(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(fmt.Sprintf("{}")))
	}))
	defer server.Close()

	cfg := aws.Config{
		Endpoint: aws.String(server.URL),
	}

	sess := unit.Session.Copy(&cfg)
	r := csm.Get()

	r.InjectHandlers(&sess.Handlers)

	c := sess.ClientConfig("id", &cfg)

	svc := mockService{
		client.New(
			*c.Config,
			metadata.ClientInfo{
				ServiceName:   "service",
				ServiceID:     "id",
				SigningName:   "signing",
				SigningRegion: "region",
				Endpoint:      server.URL,
				APIVersion:    "0",
				JSONVersion:   "1.1",
				TargetPrefix:  "prefix",
			},
			c.Handlers,
		),
	}

	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

	for i := 0; i < b.N; i++ {
		req := svc.Request(input{})
		req.Send()
	}
}

func BenchmarkWithCSMNoUDPConnection(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(fmt.Sprintf("{}")))
	}))
	defer server.Close()

	cfg := aws.Config{
		Endpoint: aws.String(server.URL),
	}

	sess := unit.Session.Copy(&cfg)
	r := csm.Get()
	r.Pause()
	r.InjectHandlers(&sess.Handlers)
	defer r.Pause()

	c := sess.ClientConfig("id", &cfg)

	svc := mockService{
		client.New(
			*c.Config,
			metadata.ClientInfo{
				ServiceName:   "service",
				ServiceID:     "id",
				SigningName:   "signing",
				SigningRegion: "region",
				Endpoint:      server.URL,
				APIVersion:    "0",
				JSONVersion:   "1.1",
				TargetPrefix:  "prefix",
			},
			c.Handlers,
		),
	}

	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

	for i := 0; i < b.N; i++ {
		req := svc.Request(input{})
		req.Send()
	}
}

func BenchmarkWithoutCSM(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(fmt.Sprintf("{}")))
	}))
	defer server.Close()

	cfg := aws.Config{
		Endpoint: aws.String(server.URL),
	}
	sess := unit.Session.Copy(&cfg)
	c := sess.ClientConfig("id", &cfg)

	svc := mockService{
		client.New(
			*c.Config,
			metadata.ClientInfo{
				ServiceName:   "service",
				ServiceID:     "id",
				SigningName:   "signing",
				SigningRegion: "region",
				Endpoint:      server.URL,
				APIVersion:    "0",
				JSONVersion:   "1.1",
				TargetPrefix:  "prefix",
			},
			c.Handlers,
		),
	}

	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

	for i := 0; i < b.N; i++ {
		req := svc.Request(input{})
		req.Send()
	}
}

func keys(m map[string]interface{}) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
