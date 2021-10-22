// +build go1.7

package s3control

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

type testParams struct {
	bucket                     string
	config                     *aws.Config
	expectedEndpoint           string
	expectedSigningName        string
	expectedSigningRegion      string
	expectedHeaderForOutpostID string
	expectedHeaderForAccountID bool
	expectedErr                string
}

// Test endpoint from outpost access point
func TestEndpoint_OutpostAccessPointARN(t *testing.T) {
	cases := map[string]testParams{
		"Outpost AccessPoint with no S3UseARNRegion flag set": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedEndpoint:           "https://s3-outposts.us-west-2.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-west-2",
			expectedHeaderForAccountID: true,
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
		"Outpost AccessPoint Cross-Region Enabled": {
			bucket: "arn:aws:s3-outposts:us-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:           "https://s3-outposts.us-east-1.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-east-1",
			expectedHeaderForAccountID: true,
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
		"Outpost AccessPoint Cross-Region Disabled": {
			bucket: "arn:aws:s3-outposts:us-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "client region does not match provided ARN region",
		},
		"Outpost AccessPoint other partition": {
			bucket: "arn:aws-cn:s3-outposts:cn-north-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedErr: "ConfigurationError: client partition does not match provided ARN partition",
		},
		"Outpost AccessPoint us-gov region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:         aws.String("us-gov-east-1"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:           "https://s3-outposts.us-gov-east-1.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-gov-east-1",
			expectedHeaderForAccountID: true,
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
		"Outpost AccessPoint with client region as Fips": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				EndpointResolver: endpoints.AwsUsGovPartition(),
				Region:           aws.String("us-gov-east-1-fips"),
			},
			expectedErr: "InvalidARNError: resource ARN not supported for FIPS region",
		},
		"Outpost AccessPoint with client Fips region and use arn region enabled": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				EndpointResolver: endpoints.AwsUsGovPartition(),
				Region:           aws.String("us-gov-east-1-fips"),
				S3UseARNRegion:   aws.Bool(true),
			},
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-gov-east-1",
			expectedEndpoint:           "https://s3-outposts.us-gov-east-1.amazonaws.com",
			expectedHeaderForAccountID: true,
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
		"Outpost AccessPoint Fips region in Arn": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1-fips:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				EndpointResolver:        endpoints.AwsUsGovPartition(),
				EnforceShouldRetryCheck: nil,
				Region:                  aws.String("us-gov-east-1-fips"),
				DisableSSL:              nil,
				HTTPClient:              nil,
				S3UseARNRegion:          aws.Bool(true),
			},
			expectedErr: "InvalidARNError: resource ARN not supported for FIPS region",
		},
		"Outpost AccessPoint Fips region with valid ARN region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				EndpointResolver: endpoints.AwsUsGovPartition(),
				Region:           aws.String("fips-us-gov-east-1"),
				S3UseARNRegion:   aws.Bool(true),
			},
			expectedEndpoint:           "https://s3-outposts.us-gov-east-1.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-gov-east-1",
			expectedHeaderForAccountID: true,
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
		"Outpost AccessPoint with DualStack": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:       aws.String("us-west-2"),
				UseDualStack: aws.Bool(true),
			},
			expectedErr: "ConfigurationError: client configured for S3 Dual-stack but is not supported with resource ARN",
		},
		"Outpost AccessPoint with Accelerate": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:          aws.String("us-west-2"),
				S3UseAccelerate: aws.Bool(true),
			},
			expectedErr: "ConfigurationError: client configured for S3 Accelerate but is not supported with resource ARN",
		},
		"Invalid outpost resource format": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "outpost resource-id not set",
		},
		"Missing access point for outpost resource": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "incomplete outpost resource type",
		},
		"access point": {
			bucket: "myaccesspoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedEndpoint:           "https://123456789012.s3-control.us-west-2.amazonaws.com",
			expectedHeaderForAccountID: true,
			expectedSigningRegion:      "us-west-2",
			expectedSigningName:        "s3",
		},
		"outpost access point with unsupported sub-resource": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:accesspoint:mybucket:object:foo",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "sub resource not supported",
		},
		"Missing outpost identifiers in outpost access point arn": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "invalid Amazon s3-outposts ARN",
		},
	}

	runValidations(t, cases)
}

// Test endpoint from outpost bucket arn
func TestEndpoint_OutpostBucketARN(t *testing.T) {
	cases := map[string]testParams{
		"Outpost Bucket with no S3UseARNRegion flag set": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedEndpoint:           "https://s3-outposts.us-west-2.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-west-2",
			expectedHeaderForOutpostID: "op-01234567890123456",
			expectedHeaderForAccountID: true,
		},
		"Outpost Bucket Cross-Region Enabled": {
			bucket: "arn:aws:s3-outposts:us-east-1:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:           "https://s3-outposts.us-east-1.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-east-1",
			expectedHeaderForOutpostID: "op-01234567890123456",
			expectedHeaderForAccountID: true,
		},
		"Outpost Bucket Cross-Region Disabled": {
			bucket: "arn:aws:s3-outposts:us-east-1:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "client region does not match provided ARN region",
		},
		"Outpost Bucket other partition": {
			bucket: "arn:aws-cn:s3-outposts:cn-north-1:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedErr: "ConfigurationError: client partition does not match provided ARN partition",
		},
		"Outpost Bucket us-gov region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				Region:         aws.String("us-gov-east-1"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:           "https://s3-outposts.us-gov-east-1.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-gov-east-1",
			expectedHeaderForOutpostID: "op-01234567890123456",
			expectedHeaderForAccountID: true,
		},
		"Outpost Bucket Fips region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				EndpointResolver: endpoints.AwsUsGovPartition(),
				Region:           aws.String("fips-us-gov-east-1"),
			},
			expectedErr: "ConfigurationError: client region does not match provided ARN region",
		},
		"Outpost Bucket Fips region in Arn": {
			bucket: "arn:aws-us-gov:s3-outposts:fips-us-gov-east-1:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				EndpointResolver:        endpoints.AwsUsGovPartition(),
				EnforceShouldRetryCheck: nil,
				Region:                  aws.String("fips-us-gov-east-1"),
				DisableSSL:              nil,
				HTTPClient:              nil,
				S3UseARNRegion:          aws.Bool(true),
			},
			expectedErr: "InvalidARNError: resource ARN not supported for FIPS region",
		},
		"Outpost Bucket Fips region with valid ARN region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				EndpointResolver: endpoints.AwsUsGovPartition(),
				Region:           aws.String("fips-us-gov-east-1"),
				S3UseARNRegion:   aws.Bool(true),
			},
			expectedEndpoint:           "https://s3-outposts.us-gov-east-1.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-gov-east-1",
			expectedHeaderForOutpostID: "op-01234567890123456",
			expectedHeaderForAccountID: true,
		},
		"Outpost Bucket with DualStack": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:bucket:mybucket",
			config: &aws.Config{
				Region:       aws.String("us-west-2"),
				UseDualStack: aws.Bool(true),
			},
			expectedErr: "ConfigurationError: client configured for S3 Dual-stack but is not supported with resource ARN",
		},
		"Outpost Bucket with Accelerate": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:          aws.String("us-west-2"),
				S3UseAccelerate: aws.Bool(true),
			},
			expectedErr: "ConfigurationError: client configured for S3 Accelerate but is not supported with resource ARN",
		},
		"Missing bucket id": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:bucket",
			config: &aws.Config{
				Region:          aws.String("us-west-2"),
				S3UseAccelerate: aws.Bool(true),
			},
			expectedErr: "invalid Amazon s3-outposts ARN",
		},
		"Invalid ARN": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:bucket:mybucket",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "invalid Amazon s3-outposts ARN, unknown resource type",
		},
	}

	runValidations(t, cases)
}

// Runs the test validation
func runValidations(t *testing.T, cases map[string]testParams) {
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			sess := unit.Session.Copy(c.config)

			svc := New(sess)
			req, _ := svc.GetBucketRequest(&GetBucketInput{
				Bucket:    &c.bucket,
				AccountId: aws.String("123456789012"),
			})

			req.Handlers.Send.Clear()
			req.Handlers.Send.PushBack(func(r *request.Request) {
				defer func() {
					r.HTTPResponse = &http.Response{
						StatusCode:    200,
						ContentLength: 0,
						Body:          ioutil.NopCloser(bytes.NewReader(nil)),
					}
				}()
				if len(c.expectedErr) != 0 {
					return
				}

				endpoint := fmt.Sprintf("%s://%s", r.HTTPRequest.URL.Scheme, r.HTTPRequest.URL.Host)
				if e, a := c.expectedEndpoint, endpoint; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}

				if e, a := c.expectedSigningName, r.ClientInfo.SigningName; c.config.Endpoint == nil && e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				if e, a := c.expectedSigningRegion, r.ClientInfo.SigningRegion; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}

				if e, a := c.expectedHeaderForOutpostID, r.HTTPRequest.Header.Get("x-amz-outpost-id"); e != a {
					if len(e) == 0 {
						t.Errorf("expected no outpost id header set, got %v", a)
					} else if len(a) == 0 {
						t.Errorf("expected outpost id header set as %v, got none", e)
					} else {
						t.Errorf("expected %v as Outpost id header value, got %v", e, a)
					}
				}

				if c.expectedHeaderForAccountID {
					if e, a := "123456789012", r.HTTPRequest.Header.Get("x-amz-account-id"); e != a {
						t.Errorf("expected x-amz-account-id header value to be %v, got %v", e, a)
					}
				}
			})

			err := req.Send()
			if len(c.expectedErr) == 0 && err != nil {
				t.Errorf("expected no error but got: %v", err)
			} else if len(c.expectedErr) != 0 && err == nil {
				t.Errorf("expected err %q, but got nil", c.expectedErr)
			} else if len(c.expectedErr) != 0 && err != nil && !strings.Contains(err.Error(), c.expectedErr) {
				t.Errorf("expected %v, got %v", c.expectedErr, err.Error())
			}
		})
	}
}

func TestCustomEndpoint_SpecialOperations(t *testing.T) {
	cases := map[string]struct {
		bucket                     string
		outpostID                  string
		config                     *aws.Config
		requestFn                  func(c *S3Control) *request.Request
		expectedEndpoint           string
		expectedSigningName        string
		expectedSigningRegion      string
		expectedHeaderForOutpostID string
		expectedErr                string
	}{
		"CreateBucketOperation": {
			bucket:    "mockBucket",
			outpostID: "op-01234567890123456",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			requestFn: func(svc *S3Control) *request.Request {
				req, _ := svc.CreateBucketRequest(&CreateBucketInput{
					Bucket:    aws.String("mockBucket"),
					OutpostId: aws.String("op-01234567890123456"),
				})
				return req
			},
			expectedEndpoint:           "https://s3-outposts.us-west-2.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-west-2",
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
		"ListRegionalBucketsOperation": {
			bucket:    "mockBucket",
			outpostID: "op-01234567890123456",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			requestFn: func(svc *S3Control) *request.Request {
				req, _ := svc.ListRegionalBucketsRequest(&ListRegionalBucketsInput{
					OutpostId: aws.String("op-01234567890123456"),
					AccountId: aws.String("123456789012"),
				})
				return req
			},
			expectedEndpoint:           "https://s3-outposts.us-west-2.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-west-2",
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
		"CreateAccessPoint bucket arn": {
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			requestFn: func(svc *S3Control) *request.Request {
				req, _ := svc.CreateAccessPointRequest(&CreateAccessPointInput{
					AccountId: aws.String("123456789012"),
					Bucket:    aws.String("arn:aws:s3:us-west-2:123456789012:bucket:mockBucket"),
					Name:      aws.String("mockName"),
				})
				return req
			},
			expectedErr: "invalid Amazon s3 ARN, unknown resource type",
		},
		"CreateAccessPoint outpost bucket arn": {
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			requestFn: func(svc *S3Control) *request.Request {
				req, _ := svc.CreateAccessPointRequest(&CreateAccessPointInput{
					AccountId: aws.String("123456789012"),
					Bucket:    aws.String("arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:bucket:mockBucket"),
					Name:      aws.String("mockName"),
				})
				return req
			},
			expectedEndpoint:           "https://s3-outposts.us-west-2.amazonaws.com",
			expectedSigningName:        "s3-outposts",
			expectedSigningRegion:      "us-west-2",
			expectedHeaderForOutpostID: "op-01234567890123456",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			sess := unit.Session.Copy(c.config)
			svc := New(sess)
			req := c.requestFn(svc)
			req.Handlers.Send.Clear()
			req.Handlers.Send.PushBack(func(r *request.Request) {
				defer func() {
					r.HTTPResponse = &http.Response{
						StatusCode:    200,
						ContentLength: 0,
						Body:          ioutil.NopCloser(bytes.NewReader(nil)),
					}
				}()
				if len(c.expectedErr) != 0 {
					return
				}

				endpoint := fmt.Sprintf("%s://%s", r.HTTPRequest.URL.Scheme, r.HTTPRequest.URL.Host)
				if e, a := c.expectedEndpoint, endpoint; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}

				if e, a := c.expectedSigningName, r.ClientInfo.SigningName; c.config.Endpoint == nil && e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				if e, a := c.expectedSigningRegion, r.ClientInfo.SigningRegion; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}

				if e, a := c.expectedHeaderForOutpostID, r.HTTPRequest.Header.Get("x-amz-outpost-id"); e != a {
					if len(e) == 0 {
						t.Errorf("expected no outpost id header set, got %v", a)
					} else if len(a) == 0 {
						t.Errorf("expected outpost id header set as %v, got none", e)
					} else {
						t.Errorf("expected %v as Outpost id header value, got %v", e, a)
					}
				}
			})

			err := req.Send()
			if len(c.expectedErr) == 0 && err != nil {
				t.Errorf("expected no error but got: %v", err)
			} else if len(c.expectedErr) != 0 && err == nil {
				t.Errorf("expected err %q, but got nil", c.expectedErr)
			} else if len(c.expectedErr) != 0 && err != nil && !strings.Contains(err.Error(), c.expectedErr) {
				t.Errorf("expected %v, got %v", c.expectedErr, err.Error())
			}
		})
	}
}

func TestInputIsNotModified(t *testing.T) {
	inputBucket := "arn:aws:s3-outposts:us-east-1:123456789012:outpost:op-01234567890123456:bucket:mybucket"
	expectedAccountID := "123456789012"
	sess := unit.Session.Copy(&aws.Config{
		Region:         aws.String("us-west-2"),
		S3UseARNRegion: aws.Bool(true),
	})

	svc := New(sess)
	params := &DeleteBucketInput{
		Bucket: aws.String(inputBucket),
	}
	req, _ := svc.DeleteBucketRequest(params)

	req.Handlers.Send.Clear()
	req.Handlers.Send.PushBack(func(r *request.Request) {
		defer func() {
			r.HTTPResponse = &http.Response{
				StatusCode:    200,
				ContentLength: 0,
				Body:          ioutil.NopCloser(bytes.NewReader(nil)),
			}
		}()
	})

	err := req.Send()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// check if req params were modified
	if e, a := *params.Bucket, inputBucket; !strings.EqualFold(e, a) {
		t.Fatalf("expected no modification for operation input, "+
			"expected %v, got %v as bucket input", e, a)
	}

	if params.AccountId != nil {
		t.Fatalf("expected original input to be unmodified, but account id was backfilled")
	}

	modifiedInput, ok := req.Params.(*DeleteBucketInput)
	if !ok {
		t.Fatalf("expected modified input of type *DeleteBucketInput")
	}

	if modifiedInput.AccountId == nil {
		t.Fatalf("expected AccountID value to be backfilled, was not")
	}

	if e, a := expectedAccountID, *modifiedInput.AccountId; !strings.EqualFold(e, a) {
		t.Fatalf("expected account id backfilled on request params to be %v, got %v", e, a)
	}

	if modifiedInput.Bucket == nil {
		t.Fatalf("expected Bucket value to be set, was not")
	}

	if e, a := "mybucket", *modifiedInput.Bucket; !strings.EqualFold(e, a) {
		t.Fatalf("expected modified input bucket name to be %v, got %v", e, a)
	}
}
