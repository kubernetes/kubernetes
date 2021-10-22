package s3_test

import (
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func TestSSECustomerKeyOverHTTPError(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{DisableSSL: aws.Bool(true)})
	req, _ := s.CopyObjectRequest(&s3.CopyObjectInput{
		Bucket:         aws.String("bucket"),
		CopySource:     aws.String("bucket/source"),
		Key:            aws.String("dest"),
		SSECustomerKey: aws.String("key"),
	})
	err := req.Build()

	if err == nil {
		t.Error("expected an error")
	}
	if e, a := "ConfigError", err.(awserr.Error).Code(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if !strings.Contains(err.(awserr.Error).Message(), "cannot send SSE keys over HTTP") {
		t.Errorf("expected error to contain 'cannot send SSE keys over HTTP', but received %s", err.(awserr.Error).Message())
	}
}

func TestCopySourceSSECustomerKeyOverHTTPError(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{DisableSSL: aws.Bool(true)})
	req, _ := s.CopyObjectRequest(&s3.CopyObjectInput{
		Bucket:                   aws.String("bucket"),
		CopySource:               aws.String("bucket/source"),
		Key:                      aws.String("dest"),
		CopySourceSSECustomerKey: aws.String("key"),
	})
	err := req.Build()

	if err == nil {
		t.Error("expected an error")
	}
	if e, a := "ConfigError", err.(awserr.Error).Code(); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if !strings.Contains(err.(awserr.Error).Message(), "cannot send SSE keys over HTTP") {
		t.Errorf("expected error to contain 'cannot send SSE keys over HTTP', but received %s", err.(awserr.Error).Message())
	}
}

func TestComputeSSEKeys(t *testing.T) {
	s := s3.New(unit.Session)
	req, _ := s.CopyObjectRequest(&s3.CopyObjectInput{
		Bucket:                   aws.String("bucket"),
		CopySource:               aws.String("bucket/source"),
		Key:                      aws.String("dest"),
		SSECustomerKey:           aws.String("key"),
		CopySourceSSECustomerKey: aws.String("key"),
	})
	err := req.Build()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if e, a := "a2V5", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "a2V5", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "PG4LipwVIkqCKLmpjKFTHQ==", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key-md5"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "PG4LipwVIkqCKLmpjKFTHQ==", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key-md5"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestComputeSSEKeysShortcircuit(t *testing.T) {
	s := s3.New(unit.Session)
	req, _ := s.CopyObjectRequest(&s3.CopyObjectInput{
		Bucket:                      aws.String("bucket"),
		CopySource:                  aws.String("bucket/source"),
		Key:                         aws.String("dest"),
		SSECustomerKey:              aws.String("key"),
		CopySourceSSECustomerKey:    aws.String("key"),
		SSECustomerKeyMD5:           aws.String("MD5"),
		CopySourceSSECustomerKeyMD5: aws.String("MD5"),
	})
	err := req.Build()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if e, a := "a2V5", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "a2V5", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "MD5", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key-md5"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "MD5", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key-md5"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestSSECustomerKeysWithSpaces(t *testing.T) {
	s := s3.New(unit.Session)
	req, _ := s.CopyObjectRequest(&s3.CopyObjectInput{
		Bucket:                   aws.String("bucket"),
		CopySource:               aws.String("bucket/source"),
		Key:                      aws.String("dest"),
		SSECustomerKey:           aws.String("   key   "),
		CopySourceSSECustomerKey: aws.String("   copykey   "),
	})
	err := req.Build()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if e, a := "ICAga2V5ICAg", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "ICAgY29weWtleSAgIA==", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "13XiUSCa6ReZ3CHtCLiJLg==", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key-md5"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
	if e, a := "MHVtfmuml539o1871Vsc6w==", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key-md5"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}
