// +build integration

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

// Searches the buckets of an account that match the prefix, and deletes
// those buckets, and all objects within. Before deleting will prompt user
// to confirm bucket should be deleted. Positive confirmation is required.
//
// Usage:
//    go run deleteBuckets.go <bucketPrefix>
func main() {
	sess := session.Must(session.NewSession())

	svc := s3.New(sess)
	buckets, err := svc.ListBuckets(&s3.ListBucketsInput{})
	if err != nil {
		panic(fmt.Sprintf("failed to list buckets, %v", err))
	}

	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "bucket prefix required")
		os.Exit(1)
	}
	bucketPrefix := os.Args[1]

	var failed bool
	for _, b := range buckets.Buckets {
		bucket := aws.StringValue(b.Name)

		if !strings.HasPrefix(bucket, bucketPrefix) {
			continue
		}

		fmt.Printf("Delete bucket %q? [y/N]: ", bucket)
		var v string
		if _, err := fmt.Scanln(&v); err != nil || !(v == "Y" || v == "y") {
			fmt.Println("\tSkipping")
			continue
		}

		fmt.Println("\tDeleting")
		if err := deleteBucket(svc, bucket); err != nil {
			fmt.Fprintf(os.Stderr, "failed to delete bucket %q, %v", bucket, err)
			failed = true
		}
	}

	if failed {
		os.Exit(1)
	}
}

func deleteBucket(svc *s3.S3, bucket string) error {
	bucketName := &bucket

	objs, err := svc.ListObjects(&s3.ListObjectsInput{Bucket: bucketName})
	if err != nil {
		return fmt.Errorf("failed to list bucket %q objects, %v", *bucketName, err)
	}

	for _, o := range objs.Contents {
		svc.DeleteObject(&s3.DeleteObjectInput{Bucket: bucketName, Key: o.Key})
	}

	uploads, err := svc.ListMultipartUploads(&s3.ListMultipartUploadsInput{Bucket: bucketName})
	if err != nil {
		return fmt.Errorf("failed to list bucket %q multipart objects, %v", *bucketName, err)
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
		return fmt.Errorf("failed to delete bucket %q, %v", *bucketName, err)
	}

	return nil
}
