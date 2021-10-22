// +build example

package main

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/cloudwatchlogs"
)

func main() {
	sess := session.Must(
		session.NewSession(&aws.Config{
			// Use a custom retryer to provide custom retry rules.
			Retryer: CustomRetryer{
				DefaultRetryer: client.DefaultRetryer{
					NumMaxRetries: client.DefaultRetryerMaxNumRetries,
				}},

			// Use the SDK's SharedCredentialsProvider directly instead of the
			// SDK's default credential chain. This ensures that the
			// application can call Config.Credentials.Expire. This  is counter
			// to the SDK's default credentials chain, which  will never reread
			// the shared credentials file.
			Credentials: credentials.NewCredentials(&credentials.SharedCredentialsProvider{
				Filename: defaults.SharedCredentialsFilename(),
				Profile:  "default",
			}),
			Region: aws.String(endpoints.UsWest2RegionID),
		}),
	)
	// Add a request handler to the AfterRetry handler stack that is used by the
	// SDK to be executed after the SDK has determined if it will retry.
	// This handler forces the SDK's Credentials to be expired, and next call to
	// Credentials.Get will attempt to refresh the credentials.
	sess.Handlers.AfterRetry.PushBack(func(req *request.Request) {
		if aerr, ok := req.Error.(awserr.RequestFailure); ok && aerr != nil {
			if aerr.Code() == "InvalidClaimException" {
				// Force the credentials to expire based on error code.  Next
				// call to Credentials.Get will attempt to refresh credentials.
				req.Config.Credentials.Expire()
			}
		}
	})

	svc := cloudwatchlogs.New(sess)

	resp, err := svc.DescribeLogGroups(&cloudwatchlogs.DescribeLogGroupsInput{})

	fmt.Println(resp, err)
}

// CustomRetryer wraps the SDK's built in DefaultRetryer adding additional
// custom features. Such as, no retry for 5xx status codes, and refresh
// credentials.
type CustomRetryer struct {
	client.DefaultRetryer
}

// ShouldRetry overrides the SDK's built in DefaultRetryer adding customization
// to not retry 5xx status codes.
func (r CustomRetryer) ShouldRetry(req *request.Request) bool {
	if req.HTTPResponse.StatusCode >= 500 {
		// Don't retry any 5xx status codes.
		return false
	}

	// Fallback to SDK's built in retry rules
	return r.DefaultRetryer.ShouldRetry(req)
}
