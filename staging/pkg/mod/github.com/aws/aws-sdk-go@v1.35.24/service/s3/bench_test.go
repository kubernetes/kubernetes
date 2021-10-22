package s3

import (
	"bytes"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

func BenchmarkPresign_GetObject(b *testing.B) {
	sess := unit.Session
	svc := New(sess)

	for i := 0; i < b.N; i++ {
		req, _ := svc.GetObjectRequest(&GetObjectInput{
			Bucket: aws.String("mock-bucket"),
			Key:    aws.String("mock-key"),
		})

		u, h, err := req.PresignRequest(15 * time.Minute)
		if err != nil {
			b.Fatalf("expect no error, got %v", err)
		}
		if len(u) == 0 {
			b.Fatalf("expect url, got none")
		}
		if len(h) != 0 {
			b.Fatalf("no signed headers, got %v", h)
		}
	}
}

func BenchmarkPresign_PutObject(b *testing.B) {
	sess := unit.Session
	svc := New(sess)

	body := make([]byte, 1024*1024*20)
	for i := 0; i < b.N; i++ {
		req, _ := svc.PutObjectRequest(&PutObjectInput{
			Bucket: aws.String("mock-bucket"),
			Key:    aws.String("mock-key"),
			Body:   bytes.NewReader(body),
		})

		u, h, err := req.PresignRequest(15 * time.Minute)
		if err != nil {
			b.Fatalf("expect no error, got %v", err)
		}
		if len(u) == 0 {
			b.Fatalf("expect url, got none")
		}
		if len(h) == 0 {
			b.Fatalf("expect signed header, got none")
		}
	}
}
