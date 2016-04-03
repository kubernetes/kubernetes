// +build integration

// Package s3_test runs integration tests for S3
package s3_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

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
	assert.NoError(t, err)

	resp, err := svc.GetObject(&s3.GetObjectInput{
		Bucket: bucketName,
		Key:    aws.String("key name"),
	})
	assert.NoError(t, err)

	b, _ := ioutil.ReadAll(resp.Body)
	assert.Equal(t, []byte("hello world"), b)
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
	assert.NoError(t, err)

	// PUT to the presigned URL with a body
	var puthttpreq *http.Request
	buf := bytes.NewReader([]byte("hello world"))
	puthttpreq, err = http.NewRequest("PUT", puturl, buf)
	assert.NoError(t, err)

	var putresp *http.Response
	putresp, err = http.DefaultClient.Do(puthttpreq)
	assert.NoError(t, err)
	assert.Equal(t, 200, putresp.StatusCode)

	// Presign a GET on the same URL
	getreq, _ := svc.GetObjectRequest(&s3.GetObjectInput{
		Bucket: bucketName,
		Key:    aws.String("presigned-key"),
	})

	var geturl string
	geturl, err = getreq.Presign(300 * time.Second)
	assert.NoError(t, err)

	// Get the body
	var getresp *http.Response
	getresp, err = http.Get(geturl)
	assert.NoError(t, err)

	var b []byte
	defer getresp.Body.Close()
	b, err = ioutil.ReadAll(getresp.Body)
	assert.Equal(t, "hello world", string(b))
}
