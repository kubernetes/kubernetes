// +build go1.7

package s3manager

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
)

// #1790 bug
func TestBatchDeleteContext(t *testing.T) {
	cases := []struct {
		objects     []BatchDeleteObject
		batchSize   int
		expected    int
		earlyCancel bool
		checkError  func(error) error
	}{
		0: {
			objects: []BatchDeleteObject{
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("1"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("2"),
						Bucket: aws.String("bucket2"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("3"),
						Bucket: aws.String("bucket3"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("4"),
						Bucket: aws.String("bucket4"),
					},
				},
			},
			batchSize:   1,
			expected:    0,
			earlyCancel: true,
			checkError: func(err error) error {
				batchErr, ok := err.(*BatchError)
				if !ok {
					return fmt.Errorf("expect BatchError, got %T, %v", err, err)
				}

				errs := batchErr.Errors
				if len(errs) != 4 {
					return fmt.Errorf("expected 1 batch errors, but received %d",
						len(errs))
				}

				for _, tempErr := range errs {
					aerr, ok := tempErr.OrigErr.(awserr.Error)
					if !ok {
						return fmt.Errorf("expect awserr.Error, got %T, %v",
							tempErr.OrigErr, tempErr.OrigErr)
					}

					if e, a := request.CanceledErrorCode, aerr.Code(); e != a {
						return fmt.Errorf("expect %q, error code, got %q", e, a)
					}
				}
				return nil
			},
		},
		1: {
			objects: []BatchDeleteObject{
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("1"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("2"),
						Bucket: aws.String("bucket2"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("3"),
						Bucket: aws.String("bucket3"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("4"),
						Bucket: aws.String("bucket4"),
					},
				},
			},
			batchSize: 1,
			expected:  4,
			checkError: func(err error) error {
				if err != nil {
					return fmt.Errorf("Expect no error, got %v", err)
				}
				return nil
			},
		},
	}

	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
		count++
	}))
	defer server.Close()

	svc := &mockS3Client{S3: buildS3SvcClient(server.URL)}
	for i, c := range cases {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if c.earlyCancel {
			cancel()
		}

		batcher := BatchDelete{
			Client:    svc,
			BatchSize: c.batchSize,
		}

		err := batcher.Delete(ctx, &DeleteObjectsIterator{Objects: c.objects})
		if terr := c.checkError(err); terr != nil {
			t.Fatalf("%d, %s", i, terr)
		}

		if count != c.expected {
			t.Errorf("Case %d: expected %d, but received %d", i, c.expected, count)
		}

		count = 0
	}
}
