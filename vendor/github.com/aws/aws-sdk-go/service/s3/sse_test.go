package s3_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/stretchr/testify/assert"
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

	assert.Error(t, err)
	assert.Equal(t, "ConfigError", err.(awserr.Error).Code())
	assert.Contains(t, err.(awserr.Error).Message(), "cannot send SSE keys over HTTP")
}

func TestCopySourceSSECustomerKeyOverHTTPError(t *testing.T) {
	s := s3.New(unit.Session, &aws.Config{DisableSSL: aws.Bool(true)})
	req, _ := s.CopyObjectRequest(&s3.CopyObjectInput{
		Bucket:     aws.String("bucket"),
		CopySource: aws.String("bucket/source"),
		Key:        aws.String("dest"),
		CopySourceSSECustomerKey: aws.String("key"),
	})
	err := req.Build()

	assert.Error(t, err)
	assert.Equal(t, "ConfigError", err.(awserr.Error).Code())
	assert.Contains(t, err.(awserr.Error).Message(), "cannot send SSE keys over HTTP")
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

	assert.NoError(t, err)
	assert.Equal(t, "a2V5", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key"))
	assert.Equal(t, "a2V5", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key"))
	assert.Equal(t, "PG4LipwVIkqCKLmpjKFTHQ==", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key-md5"))
	assert.Equal(t, "PG4LipwVIkqCKLmpjKFTHQ==", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key-md5"))
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

	assert.NoError(t, err)
	assert.Equal(t, "a2V5", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key"))
	assert.Equal(t, "a2V5", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key"))
	assert.Equal(t, "MD5", req.HTTPRequest.Header.Get("x-amz-server-side-encryption-customer-key-md5"))
	assert.Equal(t, "MD5", req.HTTPRequest.Header.Get("x-amz-copy-source-server-side-encryption-customer-key-md5"))
}
