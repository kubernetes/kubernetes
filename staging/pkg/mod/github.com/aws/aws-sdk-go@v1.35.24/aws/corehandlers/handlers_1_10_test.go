// +build go1.10

package corehandlers_test

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func TestSendHandler_HEADNoBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if e, a := "HEAD", r.Method; e != a {
			t.Errorf("expected %v method, got %v", e, a)
		}
		var buf bytes.Buffer
		io.Copy(&buf, r.Body)

		if n := buf.Len(); n != 0 {
			t.Errorf("expect empty body, got %d", n)
		}

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	svc := s3.New(unit.Session, &aws.Config{
		Endpoint:         aws.String(server.URL),
		Credentials:      credentials.AnonymousCredentials,
		S3ForcePathStyle: aws.Bool(true),
		DisableSSL:       aws.Bool(true),
	})

	req, _ := svc.HeadObjectRequest(&s3.HeadObjectInput{
		Bucket: aws.String("bucketname"),
		Key:    aws.String("keyname"),
	})

	if e, a := request.NoBody, req.HTTPRequest.Body; e != a {
		t.Fatalf("expect %T request body, got %T", e, a)
	}

	if err := req.Send(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := http.StatusOK, req.HTTPResponse.StatusCode; e != a {
		t.Errorf("expect %d status code, got %d", e, a)
	}
}
