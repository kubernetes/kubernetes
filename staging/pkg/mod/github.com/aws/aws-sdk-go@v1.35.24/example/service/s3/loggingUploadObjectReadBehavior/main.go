// +build example

package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"runtime/debug"

	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

// Usage:
//   go run -tags example  <bucket> <key> <file to upload>
//
// Example:
//   AWS_REGION=us-west-2 AWS_PROFILE=default go run . "mybucket" "10MB.file" ./10MB.file
func main() {
	sess, err := session.NewSession()
	if err != nil {
		log.Fatalf("failed to load session, %v", err)
	}

	uploader := s3manager.NewUploader(sess)

	file, err := os.Open(os.Args[3])
	if err != nil {
		log.Fatalf("failed to open file, %v", err)
	}
	defer file.Close()

	// Wrap the readSeeker with a logger that will log usage, and stack traces
	// on errors.
	readLogger := NewReadLogger(file, sess.Config.Logger)

	// Upload with read logger
	resp, err := uploader.Upload(&s3manager.UploadInput{
		Bucket: &os.Args[1],
		Key:    &os.Args[2],
		Body:   readLogger,
	}, func(u *s3manager.Uploader) {
		u.Concurrency = 1
		u.RequestOptions = append(u.RequestOptions, func(r *request.Request) {
		})
	})

	fmt.Println(resp, err)
}

// Logger is a logger use for logging the readers usage.
type Logger interface {
	Log(args ...interface{})
}

// ReadSeeker interface provides the interface for a Reader, Seeker, and ReadAt.
type ReadSeeker interface {
	io.ReadSeeker
	io.ReaderAt
}

// ReadLogger wraps an reader with logging for access.
type ReadLogger struct {
	reader ReadSeeker
	logger Logger
}

// NewReadLogger a ReadLogger that wraps the passed in ReadSeeker (Reader,
// Seeker, ReadAt) with a logger.
func NewReadLogger(r ReadSeeker, logger Logger) *ReadLogger {
	return &ReadLogger{
		reader: r,
		logger: logger,
	}
}

// Seek offsets the reader's current position for the next read.
func (s *ReadLogger) Seek(offset int64, mode int) (int64, error) {
	newOffset, err := s.reader.Seek(offset, mode)
	msg := fmt.Sprintf(
		"ReadLogger.Seek(offset:%d, mode:%d) (newOffset:%d, err:%v)",
		offset, mode, newOffset, err)
	if err != nil {
		msg += fmt.Sprintf("\n\tStack:\n%s", string(debug.Stack()))
	}

	s.logger.Log(msg)
	return newOffset, err
}

// Read attempts to read from the reader, returning the bytes read, or error.
func (s *ReadLogger) Read(b []byte) (int, error) {
	n, err := s.reader.Read(b)
	msg := fmt.Sprintf(
		"ReadLogger.Read(len(bytes):%d) (read:%d, err:%v)",
		len(b), n, err)
	if err != nil {
		msg += fmt.Sprintf("\n\tStack:\n%s", string(debug.Stack()))
	}

	s.logger.Log(msg)
	return n, err
}

// ReadAt will read the underlying reader starting at the offset.
func (s *ReadLogger) ReadAt(b []byte, offset int64) (int, error) {
	n, err := s.reader.ReadAt(b, offset)
	msg := fmt.Sprintf(
		"ReadLogger.ReadAt(len(bytes):%d, offset:%d) (read:%d, err:%v)",
		len(b), offset, n, err)
	if err != nil {
		msg += fmt.Sprintf("\n\tStack:\n%s", string(debug.Stack()))
	}

	s.logger.Log(msg)
	return n, err
}
