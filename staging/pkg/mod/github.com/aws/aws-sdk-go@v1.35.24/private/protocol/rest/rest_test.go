// +build go1.7

package rest_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
)

func TestUnsetHeaders(t *testing.T) {
	cfg := &aws.Config{Region: aws.String("us-west-2")}
	c := unit.Session.ClientConfig("testService", cfg)
	svc := client.New(
		*cfg,
		metadata.ClientInfo{
			ServiceName:   "testService",
			SigningName:   c.SigningName,
			SigningRegion: c.SigningRegion,
			Endpoint:      c.Endpoint,
			APIVersion:    "",
		},
		c.Handlers,
	)

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(rest.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(rest.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(rest.UnmarshalMetaHandler)
	op := &request.Operation{
		Name:     "test-operation",
		HTTPPath: "/",
	}

	input := &struct {
		Foo aws.JSONValue `location:"header" locationName:"x-amz-foo" type:"jsonvalue"`
		Bar aws.JSONValue `location:"header" locationName:"x-amz-bar" type:"jsonvalue"`
	}{}

	output := &struct {
		Foo aws.JSONValue `location:"header" locationName:"x-amz-foo" type:"jsonvalue"`
		Bar aws.JSONValue `location:"header" locationName:"x-amz-bar" type:"jsonvalue"`
	}{}

	req := svc.NewRequest(op, input, output)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewBuffer(nil)), Header: http.Header{}}
	req.HTTPResponse.Header.Set("X-Amz-Foo", "e30=")

	// unmarshal response
	rest.UnmarshalMeta(req)
	rest.Unmarshal(req)
	if req.Error != nil {
		t.Fatal(req.Error)
	}
}

func TestNormalizedHeaders(t *testing.T) {
	cases := map[string]struct {
		inputValues          map[string]*string
		outputValues         http.Header
		expectedInputHeaders http.Header
		expectedOutput       map[string]*string
		normalize            bool
	}{
		"non-normalized headers": {
			inputValues: map[string]*string{
				"baz": aws.String("bazValue"),
				"BAR": aws.String("barValue"),
			},
			expectedInputHeaders: http.Header{
				"X-Amz-Meta-Baz": []string{"bazValue"},
				"X-Amz-Meta-Bar": []string{"barValue"},
			},
			outputValues: http.Header{
				"X-Amz-Meta-Baz": []string{"bazValue"},
				"X-Amz-Meta-Bar": []string{"barValue"},
			},
			expectedOutput: map[string]*string{
				"Baz": aws.String("bazValue"),
				"Bar": aws.String("barValue"),
			},
		},
		"normalized headers": {
			inputValues: map[string]*string{
				"baz": aws.String("bazValue"),
				"BAR": aws.String("barValue"),
			},
			expectedInputHeaders: http.Header{
				"X-Amz-Meta-Baz": []string{"bazValue"},
				"X-Amz-Meta-Bar": []string{"barValue"},
			},
			outputValues: http.Header{
				"X-Amz-Meta-Baz": []string{"bazValue"},
				"X-Amz-Meta-Bar": []string{"barValue"},
			},
			expectedOutput: map[string]*string{
				"baz": aws.String("bazValue"),
				"bar": aws.String("barValue"),
			},
			normalize: true,
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := &aws.Config{Region: aws.String("us-west-2"), LowerCaseHeaderMaps: &tt.normalize}
			c := unit.Session.ClientConfig("testService", cfg)
			svc := client.New(
				*cfg,
				metadata.ClientInfo{
					ServiceName:   "testService",
					SigningName:   c.SigningName,
					SigningRegion: c.SigningRegion,
					Endpoint:      c.Endpoint,
					APIVersion:    "",
				},
				c.Handlers,
			)

			// Handlers
			op := &request.Operation{
				Name:     "test-operation",
				HTTPPath: "/",
			}

			input := &struct {
				Foo map[string]*string `location:"headers" locationName:"x-amz-meta-" type:"map"`
			}{
				Foo: tt.inputValues,
			}

			output := &struct {
				Foo map[string]*string `location:"headers" locationName:"x-amz-meta-" type:"map"`
			}{}

			req := svc.NewRequest(op, input, output)
			req.HTTPResponse = &http.Response{
				StatusCode: 200,
				Body:       ioutil.NopCloser(bytes.NewBuffer(nil)),
				Header:     tt.outputValues,
			}

			// Build Request
			rest.Build(req)
			if req.Error != nil {
				t.Fatal(req.Error)
			}

			if e, a := tt.expectedInputHeaders, req.HTTPRequest.Header; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, but got %v", e, a)
			}

			// unmarshal response
			rest.UnmarshalMeta(req)
			rest.Unmarshal(req)
			if req.Error != nil {
				t.Fatal(req.Error)
			}

			if e, a := tt.expectedOutput, output.Foo; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, but got %v", e, a)
			}
		})
	}
}
