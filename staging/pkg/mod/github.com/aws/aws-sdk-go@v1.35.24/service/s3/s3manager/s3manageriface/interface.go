// Package s3manageriface provides an interface for the s3manager package
package s3manageriface

import (
	"io"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

var _ DownloaderAPI = (*s3manager.Downloader)(nil)

// DownloaderAPI is the interface type for s3manager.Downloader.
type DownloaderAPI interface {
	Download(io.WriterAt, *s3.GetObjectInput, ...func(*s3manager.Downloader)) (int64, error)
	DownloadWithContext(aws.Context, io.WriterAt, *s3.GetObjectInput, ...func(*s3manager.Downloader)) (int64, error)
}

// DownloadWithIterator is the interface type for the contained method of the same name.
type DownloadWithIterator interface {
	DownloadWithIterator(aws.Context, s3manager.BatchDownloadIterator, ...func(*s3manager.Downloader)) error
}

var _ UploaderAPI = (*s3manager.Uploader)(nil)
var _ UploadWithIterator = (*s3manager.Uploader)(nil)

// UploaderAPI is the interface type for s3manager.Uploader.
type UploaderAPI interface {
	Upload(*s3manager.UploadInput, ...func(*s3manager.Uploader)) (*s3manager.UploadOutput, error)
	UploadWithContext(aws.Context, *s3manager.UploadInput, ...func(*s3manager.Uploader)) (*s3manager.UploadOutput, error)
}

// UploadWithIterator is the interface for uploading objects to S3 using the S3
// upload manager.
type UploadWithIterator interface {
	UploadWithIterator(aws.Context, s3manager.BatchUploadIterator, ...func(*s3manager.Uploader)) error
}

var _ BatchDelete = (*s3manager.BatchDelete)(nil)

// BatchDelete is the interface type for batch deleting objects from S3 using
// the S3 manager. (separated for user to compose).
type BatchDelete interface {
	Delete(aws.Context, s3manager.BatchDeleteIterator) error
}
