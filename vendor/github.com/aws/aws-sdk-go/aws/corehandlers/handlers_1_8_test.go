// +build go1.8

package corehandlers_test

import (
	"crypto/tls"
	"net/http"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/service/s3"
	"golang.org/x/net/http2"
)

func TestSendHandler_HEADNoBody(t *testing.T) {
	TLSBundleCertFile, TLSBundleKeyFile, TLSBundleCAFile, err := awstesting.CreateTLSBundleFiles()
	if err != nil {
		panic(err)
	}
	defer awstesting.CleanupTLSBundleFiles(TLSBundleCertFile, TLSBundleKeyFile, TLSBundleCAFile)

	endpoint, err := awstesting.CreateTLSServer(TLSBundleCertFile, TLSBundleKeyFile, nil)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	transport := http.DefaultTransport.(*http.Transport)
	// test server's certificate is self-signed certificate
	transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	http2.ConfigureTransport(transport)

	sess, err := session.NewSessionWithOptions(session.Options{
		Config: aws.Config{
			HTTPClient:       &http.Client{},
			Endpoint:         aws.String(endpoint),
			Region:           aws.String("mock-region"),
			Credentials:      credentials.AnonymousCredentials,
			S3ForcePathStyle: aws.Bool(true),
		},
	})

	svc := s3.New(sess)

	req, _ := svc.HeadObjectRequest(&s3.HeadObjectInput{
		Bucket: aws.String("bucketname"),
		Key:    aws.String("keyname"),
	})

	if e, a := request.NoBody, req.HTTPRequest.Body; e != a {
		t.Fatalf("expect %T request body, got %T", e, a)
	}

	err = req.Send()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := http.StatusOK, req.HTTPResponse.StatusCode; e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
}
