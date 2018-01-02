// +build example

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

func exitErrorf(msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, msg+"\n", args...)
	os.Exit(1)
}

// Will make a request to S3 for the contents of an object. If the request
// was successful, and the object was found the object's path and size will be
// printed to stdout.
//
// If the object's bucket or key does not exist a specific error message will
// be printed to stderr for the error.
//
// Any other error will be printed as an unknown error.
//
// Usage: handleServiceErrorCodes <bucket> <key>
func main() {
	if len(os.Args) < 3 {
		exitErrorf("Usage: %s <bucket> <key>", filepath.Base(os.Args[0]))
	}
	sess := session.Must(session.NewSession())

	svc := s3.New(sess)
	resp, err := svc.GetObject(&s3.GetObjectInput{
		Bucket: aws.String(os.Args[1]),
		Key:    aws.String(os.Args[2]),
	})

	if err != nil {
		// Casting to the awserr.Error type will allow you to inspect the error
		// code returned by the service in code. The error code can be used
		// to switch on context specific functionality. In this case a context
		// specific error message is printed to the user based on the bucket
		// and key existing.
		//
		// For information on other S3 API error codes see:
		// http://docs.aws.amazon.com/AmazonS3/latest/API/ErrorResponses.html
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case s3.ErrCodeNoSuchBucket:
				exitErrorf("bucket %s does not exist", os.Args[1])
			case s3.ErrCodeNoSuchKey:
				exitErrorf("object with key %s does not exist in bucket %s", os.Args[2], os.Args[1])
			}
		}
		exitErrorf("unknown error occurred, %v", err)
	}
	defer resp.Body.Close()

	fmt.Printf("s3://%s/%s exists. size: %d\n", os.Args[1], os.Args[2],
		aws.Int64Value(resp.ContentLength))
}
