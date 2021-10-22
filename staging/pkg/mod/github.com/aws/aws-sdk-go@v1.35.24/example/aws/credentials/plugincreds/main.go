// +build example,go18

package main

import (
	"context"
	"fmt"
	"os"
	"plugin"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials/plugincreds"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

// Example application which loads a Go Plugin file, and uses the credential
// provider defined within the plugin to get credentials for making a S3
// request.
//
// The example will derive the bucket's region automatically if a AWS_REGION
// environment variable is not defined.
//
// Build:
//   go build -tags example -o myApp main.go
//
// Usage:
//   ./myApp <compiled plugin> <bucket> <object key>
func main() {
	if len(os.Args) < 4 {
		exitErrorf("Usage: myApp <compiled plugin>, <bucket> <object key>")
	}

	pluginFilename := os.Args[1]
	bucket := os.Args[2]
	key := os.Args[3]

	// Open plugin, and load it into the process.
	p, err := plugin.Open(pluginFilename)
	if err != nil {
		exitErrorf("failed to open plugin, %s, %v", pluginFilename, err)
	}

	// Create a new Credentials value which will source the provider's Retrieve
	// and IsExpired functions from the plugin.
	creds, err := plugincreds.NewCredentials(p)
	if err != nil {
		exitErrorf("failed to load plugin provider, %v", err)
	}

	// Example to configure a Session with the newly created credentials that
	// will be sourced using the plugin's functionality.
	sess := session.Must(session.NewSession(&aws.Config{
		Credentials: creds,
	}))

	// If the region is not available attempt to derive the bucket's region
	// from a query to S3 for the bucket's metadata
	region := aws.StringValue(sess.Config.Region)
	if len(region) == 0 {
		region, err = s3manager.GetBucketRegion(context.Background(), sess, bucket, endpoints.UsEast1RegionID)
		if err != nil {
			exitErrorf("failed to get bucket region, %v", err)
		}
	}

	// Create the S3 service client for the target region
	svc := s3.New(sess, aws.NewConfig().WithRegion(region))

	// Get the object's details
	result, err := svc.HeadObject(&s3.HeadObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	fmt.Println(result, err)
}

func exitErrorf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
