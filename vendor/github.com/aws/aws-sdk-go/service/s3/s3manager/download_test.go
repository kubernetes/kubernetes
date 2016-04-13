package s3manager_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

func dlLoggingSvc(data []byte) (*s3.S3, *[]string, *[]string) {
	var m sync.Mutex
	names := []string{}
	ranges := []string{}

	svc := s3.New(unit.Session)
	svc.Handlers.Send.Clear()
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		m.Lock()
		defer m.Unlock()

		names = append(names, r.Operation.Name)
		ranges = append(ranges, *r.Params.(*s3.GetObjectInput).Range)

		rerng := regexp.MustCompile(`bytes=(\d+)-(\d+)`)
		rng := rerng.FindStringSubmatch(r.HTTPRequest.Header.Get("Range"))
		start, _ := strconv.ParseInt(rng[1], 10, 64)
		fin, _ := strconv.ParseInt(rng[2], 10, 64)
		fin++

		if fin > int64(len(data)) {
			fin = int64(len(data))
		}

		bodyBytes := data[start:fin]
		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Body:       ioutil.NopCloser(bytes.NewReader(bodyBytes)),
			Header:     http.Header{},
		}
		r.HTTPResponse.Header.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d",
			start, fin, len(data)))
		r.HTTPResponse.Header.Set("Content-Length", fmt.Sprintf("%d", len(bodyBytes)))
	})

	return svc, &names, &ranges
}

func dlLoggingSvcNoChunk(data []byte) (*s3.S3, *[]string) {
	var m sync.Mutex
	names := []string{}

	svc := s3.New(unit.Session)
	svc.Handlers.Send.Clear()
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		m.Lock()
		defer m.Unlock()

		names = append(names, r.Operation.Name)

		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Body:       ioutil.NopCloser(bytes.NewReader(data[:])),
			Header:     http.Header{},
		}
		r.HTTPResponse.Header.Set("Content-Length", fmt.Sprintf("%d", len(data)))
	})

	return svc, &names
}

func TestDownloadOrder(t *testing.T) {
	s, names, ranges := dlLoggingSvc(buf12MB)

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
	})
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	assert.Nil(t, err)
	assert.Equal(t, int64(len(buf12MB)), n)
	assert.Equal(t, []string{"GetObject", "GetObject", "GetObject"}, *names)
	assert.Equal(t, []string{"bytes=0-5242879", "bytes=5242880-10485759", "bytes=10485760-15728639"}, *ranges)

	count := 0
	for _, b := range w.Bytes() {
		count += int(b)
	}
	assert.Equal(t, 0, count)
}

func TestDownloadZero(t *testing.T) {
	s, names, ranges := dlLoggingSvc([]byte{})

	d := s3manager.NewDownloaderWithClient(s)
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	assert.Nil(t, err)
	assert.Equal(t, int64(0), n)
	assert.Equal(t, []string{"GetObject"}, *names)
	assert.Equal(t, []string{"bytes=0-5242879"}, *ranges)
}

func TestDownloadSetPartSize(t *testing.T) {
	s, names, ranges := dlLoggingSvc([]byte{1, 2, 3})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
		d.PartSize = 1
	})
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	assert.Nil(t, err)
	assert.Equal(t, int64(3), n)
	assert.Equal(t, []string{"GetObject", "GetObject", "GetObject"}, *names)
	assert.Equal(t, []string{"bytes=0-0", "bytes=1-1", "bytes=2-2"}, *ranges)
	assert.Equal(t, []byte{1, 2, 3}, w.Bytes())
}

func TestDownloadError(t *testing.T) {
	s, names, _ := dlLoggingSvc([]byte{1, 2, 3})

	num := 0
	s.Handlers.Send.PushBack(func(r *request.Request) {
		num++
		if num > 1 {
			r.HTTPResponse.StatusCode = 400
			r.HTTPResponse.Body = ioutil.NopCloser(bytes.NewReader([]byte{}))
		}
	})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
		d.PartSize = 1
	})
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	assert.NotNil(t, err)
	assert.Equal(t, int64(1), n)
	assert.Equal(t, []string{"GetObject", "GetObject"}, *names)
	assert.Equal(t, []byte{1}, w.Bytes())
}

func TestDownloadNonChunk(t *testing.T) {
	s, names := dlLoggingSvcNoChunk(buf2MB)

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
	})
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	assert.Nil(t, err)
	assert.Equal(t, int64(len(buf2MB)), n)
	assert.Equal(t, []string{"GetObject"}, *names)

	count := 0
	for _, b := range w.Bytes() {
		count += int(b)
	}
	assert.Equal(t, 0, count)
}
