// +build integration

// Package s3manager provides integration tests for the service/s3/s3manager package
package s3manager

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/integration"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

var integBuf12MB = make([]byte, 1024*1024*12)
var integMD512MB = fmt.Sprintf("%x", md5.Sum(integBuf12MB))
var bucketName *string

func TestMain(m *testing.M) {
	if err := setup(); err != nil {
		panic(fmt.Sprintf("failed to setup integration test, %v", err))
	}

	var result int

	defer func() {
		if err := teardown(); err != nil {
			fmt.Fprintf(os.Stderr, "teardown failed, %v", err)
		}
		if r := recover(); r != nil {
			fmt.Println("S3Manager integration test hit a panic,", r)
			result = 1
		}
		os.Exit(result)
	}()

	result = m.Run()
}

func setup() error {
	svc := s3.New(integration.Session)

	// Create a bucket for testing
	bucketName = aws.String(
		fmt.Sprintf("aws-sdk-go-integration-%d-%s", time.Now().Unix(), integration.UniqueID()))

	_, err := svc.CreateBucket(&s3.CreateBucketInput{Bucket: bucketName})
	if err != nil {
		return fmt.Errorf("failed to create bucket %q, %v", *bucketName, err)
	}

	err = svc.WaitUntilBucketExists(&s3.HeadBucketInput{Bucket: bucketName})
	if err != nil {
		return fmt.Errorf("failed to wait for bucket %q to exist, %v", bucketName, err)
	}

	return nil
}

// Delete the bucket
func teardown() error {
	svc := s3.New(integration.Session)

	objs, err := svc.ListObjects(&s3.ListObjectsInput{Bucket: bucketName})
	if err != nil {
		return fmt.Errorf("failed to list bucket %q objects, %v", bucketName, err)
	}

	for _, o := range objs.Contents {
		svc.DeleteObject(&s3.DeleteObjectInput{Bucket: bucketName, Key: o.Key})
	}

	uploads, err := svc.ListMultipartUploads(&s3.ListMultipartUploadsInput{Bucket: bucketName})
	if err != nil {
		return fmt.Errorf("failed to list bucket %q multipart objects, %v", bucketName, err)
	}

	for _, u := range uploads.Uploads {
		svc.AbortMultipartUpload(&s3.AbortMultipartUploadInput{
			Bucket:   bucketName,
			Key:      u.Key,
			UploadId: u.UploadId,
		})
	}

	_, err = svc.DeleteBucket(&s3.DeleteBucketInput{Bucket: bucketName})
	if err != nil {
		return fmt.Errorf("failed to delete bucket %q, %v", bucketName, err)
	}

	return nil
}

type dlwriter struct {
	buf []byte
}

func newDLWriter(size int) *dlwriter {
	return &dlwriter{buf: make([]byte, size)}
}

func (d dlwriter) WriteAt(p []byte, pos int64) (n int, err error) {
	if pos > int64(len(d.buf)) {
		return 0, io.EOF
	}

	written := 0
	for i, b := range p {
		if i >= len(d.buf) {
			break
		}
		d.buf[pos+int64(i)] = b
		written++
	}
	return written, nil
}

func validate(t *testing.T, key string, md5value string) {
	mgr := s3manager.NewDownloader(integration.Session)
	params := &s3.GetObjectInput{Bucket: bucketName, Key: &key}

	w := newDLWriter(1024 * 1024 * 20)
	n, err := mgr.Download(w, params)
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := md5value, fmt.Sprintf("%x", md5.Sum(w.buf[0:n])); e != a {
		t.Errorf("expect %s md5 value, got %s", e, a)
	}
}

func TestUploadConcurrently(t *testing.T) {
	key := "12mb-1"
	mgr := s3manager.NewUploader(integration.Session)
	out, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: bucketName,
		Key:    &key,
		Body:   bytes.NewReader(integBuf12MB),
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if len(out.UploadID) == 0 {
		t.Errorf("expect upload ID but was empty")
	}

	re := regexp.MustCompile(`^https?://.+/` + key + `$`)
	if e, a := re.String(), out.Location; !re.MatchString(a) {
		t.Errorf("expect %s to match URL regexp %q, did not", e, a)
	}

	validate(t, key, integMD512MB)
}

func TestUploadFailCleanup(t *testing.T) {
	svc := s3.New(integration.Session)

	// Break checksum on 2nd part so it fails
	part := 0
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		if r.Operation.Name == "UploadPart" {
			if part == 1 {
				r.HTTPRequest.Header.Set("X-Amz-Content-Sha256", "000")
			}
			part++
		}
	})

	key := "12mb-leave"
	mgr := s3manager.NewUploaderWithClient(svc, func(u *s3manager.Uploader) {
		u.LeavePartsOnError = false
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: bucketName,
		Key:    &key,
		Body:   bytes.NewReader(integBuf12MB),
	})
	if err == nil {
		t.Fatalf("expect error, but did not get one")
	}

	aerr := err.(awserr.Error)
	if e, a := "MissingRegion", aerr.Code(); strings.Contains(a, e) {
		t.Errorf("expect %q to not be in error code %q", e, a)
	}

	uploadID := ""
	merr := err.(s3manager.MultiUploadFailure)
	if uploadID = merr.UploadID(); len(uploadID) == 0 {
		t.Errorf("expect upload ID to not be empty, but was")
	}

	_, err = svc.ListParts(&s3.ListPartsInput{
		Bucket: bucketName, Key: &key, UploadId: &uploadID,
	})
	if err == nil {
		t.Errorf("expect error for list parts, but got none")
	}
}
