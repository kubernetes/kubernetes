// +build go1.7

package s3

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

func TestEndpoint(t *testing.T) {
	cases := map[string]struct {
		bucket                string
		config                *aws.Config
		expectedEndpoint      string
		expectedSigningName   string
		expectedSigningRegion string
		expectedErr           string
	}{
		"Outpost AccessPoint with no S3UseARNRegion flag set": {
			bucket: "arn:aws:s3-outposts:us-west-2:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedEndpoint:      "https://myaccesspoint-123456789012.op-01234567890123456.s3-outposts.us-west-2.amazonaws.com",
			expectedSigningName:   "s3-outposts",
			expectedSigningRegion: "us-west-2",
		},
		"Outpost AccessPoint Cross-Region Enabled": {
			bucket: "arn:aws:s3-outposts:us-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:      "https://myaccesspoint-123456789012.op-01234567890123456.s3-outposts.us-east-1.amazonaws.com",
			expectedSigningName:   "s3-outposts",
			expectedSigningRegion: "us-east-1",
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
		"Outpost AccessPoint cn partition": {
			bucket: "arn:aws-cn:s3-outposts:cn-north-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region: aws.String("cn-north-1"),
			},
			expectedEndpoint:      "https://myaccesspoint-123456789012.op-01234567890123456.s3-outposts.cn-north-1.amazonaws.com.cn",
			expectedSigningName:   "s3-outposts",
			expectedSigningRegion: "cn-north-1",
		},
		"Outpost AccessPoint us-gov region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				Region:         aws.String("us-gov-east-1"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:      "https://myaccesspoint-123456789012.op-01234567890123456.s3-outposts.us-gov-east-1.amazonaws.com",
			expectedSigningName:   "s3-outposts",
			expectedSigningRegion: "us-gov-east-1",
		},
		"Outpost AccessPoint Fips region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				EndpointResolver: endpoints.AwsUsGovPartition(),
				Region:           aws.String("fips-us-gov-east-1"),
			},
			expectedErr: "ConfigurationError: client region does not match provided ARN region",
		},
		"Outpost AccessPoint Fips region in Arn": {
			bucket: "arn:aws-us-gov:s3-outposts:fips-us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
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
		"Outpost AccessPoint Fips region with valid ARN region": {
			bucket: "arn:aws-us-gov:s3-outposts:us-gov-east-1:123456789012:outpost:op-01234567890123456:accesspoint:myaccesspoint",
			config: &aws.Config{
				EndpointResolver: endpoints.AwsUsGovPartition(),
				Region:           aws.String("fips-us-gov-east-1"),
				S3UseARNRegion:   aws.Bool(true),
			},
			expectedEndpoint:      "https://myaccesspoint-123456789012.op-01234567890123456.s3-outposts.us-gov-east-1.amazonaws.com",
			expectedSigningName:   "s3-outposts",
			expectedSigningRegion: "us-gov-east-1",
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
		"AccessPoint": {
			bucket: "arn:aws:s3:us-west-2:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint.us-west-2.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"AccessPoint slash delimiter": {
			bucket: "arn:aws:s3:us-west-2:123456789012:accesspoint/myendpoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint.us-west-2.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"AccessPoint other partition": {
			bucket: "arn:aws-cn:s3:cn-north-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region: aws.String("cn-north-1"),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint.cn-north-1.amazonaws.com.cn",
			expectedSigningName:   "s3",
			expectedSigningRegion: "cn-north-1",
		},
		"AccessPoint Cross-Region Disabled": {
			bucket: "arn:aws:s3:ap-south-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
			},
			expectedErr: "client region does not match provided ARN region",
		},
		"AccessPoint Cross-Region Enabled": {
			bucket: "arn:aws:s3:ap-south-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint.ap-south-1.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "ap-south-1",
		},
		"AccessPoint us-east-1": {
			bucket: "arn:aws:s3:us-east-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:         aws.String("us-east-1"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint.us-east-1.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-east-1",
		},
		"AccessPoint us-east-1 cross region": {
			bucket: "arn:aws:s3:us-east-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint.us-east-1.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-east-1",
		},
		"AccessPoint Cross-Partition not supported": {
			bucket: "arn:aws-cn:s3:cn-north-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:         aws.String("us-west-2"),
				UseDualStack:   aws.Bool(true),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedErr: "client partition does not match provided ARN partition",
		},
		"AccessPoint DualStack": {
			bucket: "arn:aws:s3:us-west-2:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:       aws.String("us-west-2"),
				UseDualStack: aws.Bool(true),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint.dualstack.us-west-2.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"AccessPoint FIPS same region with cross region disabled": {
			bucket: "arn:aws-us-gov:s3:us-gov-west-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region: aws.String("fips-us-gov-west-1"),
				EndpointResolver: endpoints.ResolverFunc(
					func(service, region string, opts ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
						switch region {
						case "fips-us-gov-west-1":
							return endpoints.ResolvedEndpoint{
								URL:           "s3-fips.us-gov-west-1.amazonaws.com",
								PartitionID:   "aws-us-gov",
								SigningRegion: "us-gov-west-1",
								SigningName:   service,
								SigningMethod: "s3v4",
							}, nil
						}
						return endpoints.ResolvedEndpoint{}, nil
					}),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint-fips.us-gov-west-1.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-gov-west-1",
		},
		"AccessPoint FIPS same region with cross region enabled": {
			bucket: "arn:aws-us-gov:s3:us-gov-west-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region: aws.String("fips-us-gov-west-1"),
				EndpointResolver: endpoints.ResolverFunc(
					func(service, region string, opts ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
						switch region {
						case "fips-us-gov-west-1":
							return endpoints.ResolvedEndpoint{
								URL:           "s3-fips.us-gov-west-1.amazonaws.com",
								PartitionID:   "aws-us-gov",
								SigningRegion: "us-gov-west-1",
								SigningName:   service,
								SigningMethod: "s3v4",
							}, nil
						}
						return endpoints.ResolvedEndpoint{}, nil
					}),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedEndpoint:      "https://myendpoint-123456789012.s3-accesspoint-fips.us-gov-west-1.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-gov-west-1",
		},
		"AccessPoint FIPS cross region not supported": {
			bucket: "arn:aws-us-gov:s3:us-gov-east-1:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:         aws.String("fips-us-gov-west-1"),
				S3UseARNRegion: aws.Bool(true),
			},
			expectedErr: "client configured for FIPS",
		},
		"AccessPoint Accelerate not supported": {
			bucket: "arn:aws:s3:us-west-2:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:          aws.String("us-west-2"),
				S3UseAccelerate: aws.Bool(true),
			},
			expectedErr: "client configured for S3 Accelerate",
		},
		"Custom Resolver Without PartitionID in ClientInfo": {
			bucket: "arn:aws:s3:us-west-2:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region: aws.String("us-west-2"),
				EndpointResolver: endpoints.ResolverFunc(
					func(service, region string, opts ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
						switch region {
						case "us-west-2":
							return endpoints.ResolvedEndpoint{
								URL:           "s3.us-west-2.amazonaws.com",
								SigningRegion: "us-west-2",
								SigningName:   service,
								SigningMethod: "s3v4",
							}, nil
						}
						return endpoints.ResolvedEndpoint{}, nil
					}),
			},
			expectedErr: "client partition does not match provided ARN partition",
		},
		"Custom Resolver Without PartitionID in Cross-Region Target": {
			bucket: "arn:aws:s3:us-west-2:123456789012:accesspoint:myendpoint",
			config: &aws.Config{
				Region:         aws.String("us-east-1"),
				S3UseARNRegion: aws.Bool(true),
				EndpointResolver: endpoints.ResolverFunc(
					func(service, region string, opts ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
						switch region {
						case "us-west-2":
							return endpoints.ResolvedEndpoint{
								URL:           "s3.us-west-2.amazonaws.com",
								PartitionID:   "aws",
								SigningRegion: "us-west-2",
								SigningName:   service,
								SigningMethod: "s3v4",
							}, nil
						case "us-east-1":
							return endpoints.ResolvedEndpoint{
								URL:           "s3.us-east-1.amazonaws.com",
								SigningRegion: "us-east-1",
								SigningName:   service,
								SigningMethod: "s3v4",
							}, nil
						}

						return endpoints.ResolvedEndpoint{}, nil
					}),
			},
			expectedErr: "client partition does not match provided ARN partition",
		},
		"bucket host-style": {
			bucket:                "mock-bucket",
			config:                &aws.Config{Region: aws.String("us-west-2")},
			expectedEndpoint:      "https://mock-bucket.s3.us-west-2.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"bucket path-style": {
			bucket: "mock-bucket",
			config: &aws.Config{
				Region:           aws.String("us-west-2"),
				S3ForcePathStyle: aws.Bool(true),
			},
			expectedEndpoint:      "https://s3.us-west-2.amazonaws.com",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"bucket host-style endpoint with default port": {
			bucket: "mock-bucket",
			config: &aws.Config{
				Region:   aws.String("us-west-2"),
				Endpoint: aws.String("https://s3.us-west-2.amazonaws.com:443"),
			},
			expectedEndpoint:      "https://mock-bucket.s3.us-west-2.amazonaws.com:443",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"bucket host-style endpoint with non-default port": {
			bucket: "mock-bucket",
			config: &aws.Config{
				Region:   aws.String("us-west-2"),
				Endpoint: aws.String("https://s3.us-west-2.amazonaws.com:8443"),
			},
			expectedEndpoint:      "https://mock-bucket.s3.us-west-2.amazonaws.com:8443",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"bucket path-style endpoint with default port": {
			bucket: "mock-bucket",
			config: &aws.Config{
				Region:           aws.String("us-west-2"),
				Endpoint:         aws.String("https://s3.us-west-2.amazonaws.com:443"),
				S3ForcePathStyle: aws.Bool(true),
			},
			expectedEndpoint:      "https://s3.us-west-2.amazonaws.com:443",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
		"bucket path-style endpoint with non-default port": {
			bucket: "mock-bucket",
			config: &aws.Config{
				Region:           aws.String("us-west-2"),
				Endpoint:         aws.String("https://s3.us-west-2.amazonaws.com:8443"),
				S3ForcePathStyle: aws.Bool(true),
			},
			expectedEndpoint:      "https://s3.us-west-2.amazonaws.com:8443",
			expectedSigningName:   "s3",
			expectedSigningRegion: "us-west-2",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			if strings.EqualFold("az", name) {
				fmt.Print()
			}

			sess := unit.Session.Copy(c.config)

			svc := New(sess)
			req, _ := svc.GetObjectRequest(&GetObjectInput{
				Bucket: &c.bucket,
				Key:    aws.String("testkey"),
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
