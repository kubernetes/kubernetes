// +build go1.6

package s3_test

import (
	"bytes"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func TestAdd100Continue_Added(t *testing.T) {
	svc := s3.New(unit.Session)
	r, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("dest"),
		Body:   bytes.NewReader(make([]byte, 1024*1024*5)),
	})

	err := r.Sign()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if e, a := "100-Continue", r.HTTPRequest.Header.Get("Expect"); e != a {
		t.Errorf("expected %s, but received %s", e, a)
	}
}

func TestAdd100Continue_SkipDisabled(t *testing.T) {
	svc := s3.New(unit.Session, aws.NewConfig().WithS3Disable100Continue(true))
	r, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("dest"),
		Body:   bytes.NewReader(make([]byte, 1024*1024*5)),
	})

	err := r.Sign()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if r.HTTPRequest.Header.Get("Expect") != "" {
		t.Errorf("expected empty value, but received %s", r.HTTPRequest.Header.Get("Expect"))
	}
}

func TestAdd100Continue_SkipNonPUT(t *testing.T) {
	svc := s3.New(unit.Session)
	r, _ := svc.GetObjectRequest(&s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("dest"),
	})

	err := r.Sign()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if r.HTTPRequest.Header.Get("Expect") != "" {
		t.Errorf("expected empty value, but received %s", r.HTTPRequest.Header.Get("Expect"))
	}
}

func TestAdd100Continue_SkipTooSmall(t *testing.T) {
	svc := s3.New(unit.Session)
	r, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("dest"),
		Body:   bytes.NewReader(make([]byte, 1024*1024*1)),
	})

	err := r.Sign()

	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if r.HTTPRequest.Header.Get("Expect") != "" {
		t.Errorf("expected empty value, but received %s", r.HTTPRequest.Header.Get("Expect"))
	}
}
