package s3manager_test

import (
	"bytes"
	"fmt"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

// ExampleNewUploader_overrideReadSeekerProvider gives an example
// on a custom ReadSeekerWriteToProvider can be provided to Uploader
// to define how parts will be buffered in memory.
func ExampleNewUploader_overrideReadSeekerProvider() {
	sess := session.Must(session.NewSession())

	uploader := s3manager.NewUploader(sess, func(u *s3manager.Uploader) {
		// Define a strategy that will buffer 25 MiB in memory
		u.BufferProvider = s3manager.NewBufferedReadSeekerWriteToPool(25 * 1024 * 1024)
	})

	_, err := uploader.Upload(&s3manager.UploadInput{
		Bucket: aws.String("examplebucket"),
		Key:    aws.String("largeobject"),
		Body:   bytes.NewReader([]byte("large_multi_part_upload")),
	})
	if err != nil {
		fmt.Println(err.Error())
	}
}

// ExampleNewUploader_overrideTransport gives an example
// on how to override the default HTTP transport. This can
// be used to tune timeouts such as response headers, or
// write / read buffer usage when writing or reading respectively
// from the net/http transport.
func ExampleNewUploader_overrideTransport() {
	// Create Transport
	tr := &http.Transport{
		ResponseHeaderTimeout: 1 * time.Second,
		// WriteBufferSize: 1024*1024 // Go 1.13
		// ReadBufferSize: 1024*1024 // Go 1.13
	}

	sess := session.Must(session.NewSession(&aws.Config{
		HTTPClient: &http.Client{Transport: tr},
	}))

	uploader := s3manager.NewUploader(sess)

	_, err := uploader.Upload(&s3manager.UploadInput{
		Bucket: aws.String("examplebucket"),
		Key:    aws.String("largeobject"),
		Body:   bytes.NewReader([]byte("large_multi_part_upload")),
	})
	if err != nil {
		fmt.Println(err.Error())
	}
}
