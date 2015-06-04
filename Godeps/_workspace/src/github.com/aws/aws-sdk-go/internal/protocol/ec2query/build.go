// Package ec2query provides serialisation of AWS EC2 requests and responses.
package ec2query

//go:generate go run ../../fixtures/protocol/generate.go ../../fixtures/protocol/input/ec2.json build_test.go

import (
	"net/url"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/internal/apierr"
	"github.com/aws/aws-sdk-go/internal/protocol/query/queryutil"
)

// Build builds a request for the EC2 protocol.
func Build(r *aws.Request) {
	body := url.Values{
		"Action":  {r.Operation.Name},
		"Version": {r.Service.APIVersion},
	}
	if err := queryutil.Parse(body, r.Params, true); err != nil {
		r.Error = apierr.New("Marshal", "failed encoding EC2 Query request", err)
	}

	if r.ExpireTime == 0 {
		r.HTTPRequest.Method = "POST"
		r.HTTPRequest.Header.Set("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
		r.SetBufferBody([]byte(body.Encode()))
	} else { // This is a pre-signed request
		r.HTTPRequest.Method = "GET"
		r.HTTPRequest.URL.RawQuery = body.Encode()
	}
}
