// +build integration

// Package s3_test runs integration tests for S3
package s3_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/service/s3"
)

var bucketName *string
var svc *s3.S3

func TestMain(m *testing.M) {
	setup()
	defer teardown() // only called if we panic
	result := m.Run()
	teardown()
	os.Exit(result)
}

// Create a bucket for testing
func setup() {
	svc = s3.New(integration.Session)
	bucketName = aws.String(
		fmt.Sprintf("aws-sdk-go-integration-%d-%s", time.Now().Unix(), integration.UniqueID()))

	for i := 0; i < 10; i++ {
		_, err := svc.CreateBucket(&s3.CreateBucketInput{Bucket: bucketName})
		if err == nil {
			break
		}
	}

	for {
		_, err := svc.HeadBucket(&s3.HeadBucketInput{Bucket: bucketName})
		if err == nil {
			break
		}
		time.Sleep(1 * time.Second)
	}
}

// Delete the bucket
func teardown() {
	resp, _ := svc.ListObjects(&s3.ListObjectsInput{Bucket: bucketName})
	for _, o := range resp.Contents {
		svc.DeleteObject(&s3.DeleteObjectInput{Bucket: bucketName, Key: o.Key})
	}
	svc.DeleteBucket(&s3.DeleteBucketInput{Bucket: bucketName})
}

func TestWriteToObject(t *testing.T) {
	_, err := svc.PutObject(&s3.PutObjectInput{
		Bucket: bucketName,
		Key:    aws.String("key name"),
		Body:   bytes.NewReader([]byte("hello world")),
	})
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	resp, err := svc.GetObject(&s3.GetObjectInput{
		Bucket: bucketName,
		Key:    aws.String("key name"),
	})
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	b, _ := ioutil.ReadAll(resp.Body)
	if e, a := []byte("hello world"), b; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestPresignedGetPut(t *testing.T) {
	putreq, _ := svc.PutObjectRequest(&s3.PutObjectInput{
		Bucket: bucketName,
		Key:    aws.String("presigned-key"),
	})
	var err error

	// Presign a PUT request
	var puturl string
	puturl, err = putreq.Presign(300 * time.Second)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	// PUT to the presigned URL with a body
	var puthttpreq *http.Request
	buf := bytes.NewReader([]byte("hello world"))
	puthttpreq, err = http.NewRequest("PUT", puturl, buf)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	var putresp *http.Response
	putresp, err = http.DefaultClient.Do(puthttpreq)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := 200, putresp.StatusCode; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	// Presign a GET on the same URL
	getreq, _ := svc.GetObjectRequest(&s3.GetObjectInput{
		Bucket: bucketName,
		Key:    aws.String("presigned-key"),
	})

	var geturl string
	geturl, err = getreq.Presign(300 * time.Second)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	// Get the body
	var getresp *http.Response
	getresp, err = http.Get(geturl)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	var b []byte
	defer getresp.Body.Close()
	b, err = ioutil.ReadAll(getresp.Body)
	if e, a := "hello world", string(b); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}
