package s3manager_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"github.com/stretchr/testify/assert"
)

var emptyList = []string{}

func val(i interface{}, s string) interface{} {
	v, err := awsutil.ValuesAtPath(i, s)
	if err != nil || len(v) == 0 {
		return nil
	}
	if _, ok := v[0].(io.Reader); ok {
		return v[0]
	}

	if rv := reflect.ValueOf(v[0]); rv.Kind() == reflect.Ptr {
		return rv.Elem().Interface()
	}

	return v[0]
}

func contains(src []string, s string) bool {
	for _, v := range src {
		if s == v {
			return true
		}
	}
	return false
}

func loggingSvc(ignoreOps []string) (*s3.S3, *[]string, *[]interface{}) {
	var m sync.Mutex
	partNum := 0
	names := []string{}
	params := []interface{}{}
	svc := s3.New(unit.Session)
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.UnmarshalError.Clear()
	svc.Handlers.Send.Clear()
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		m.Lock()
		defer m.Unlock()

		if !contains(ignoreOps, r.Operation.Name) {
			names = append(names, r.Operation.Name)
			params = append(params, r.Params)
		}

		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
		}

		switch data := r.Data.(type) {
		case *s3.CreateMultipartUploadOutput:
			data.UploadId = aws.String("UPLOAD-ID")
		case *s3.UploadPartOutput:
			partNum++
			data.ETag = aws.String(fmt.Sprintf("ETAG%d", partNum))
		case *s3.CompleteMultipartUploadOutput:
			data.Location = aws.String("https://location")
			data.VersionId = aws.String("VERSION-ID")
		case *s3.PutObjectOutput:
			data.VersionId = aws.String("VERSION-ID")
		}
	})

	return svc, &names, &params
}

func buflen(i interface{}) int {
	r := i.(io.Reader)
	b, _ := ioutil.ReadAll(r)
	return len(b)
}

func TestUploadOrderMulti(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	u := s3manager.NewUploaderWithClient(s)

	resp, err := u.Upload(&s3manager.UploadInput{
		Bucket:               aws.String("Bucket"),
		Key:                  aws.String("Key"),
		Body:                 bytes.NewReader(buf12MB),
		ServerSideEncryption: aws.String("AES256"),
		ContentType:          aws.String("content/type"),
	})

	assert.NoError(t, err)
	assert.Equal(t, []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "UploadPart", "CompleteMultipartUpload"}, *ops)
	assert.Equal(t, "https://location", resp.Location)
	assert.Equal(t, "UPLOAD-ID", resp.UploadID)
	assert.Equal(t, aws.String("VERSION-ID"), resp.VersionID)

	// Validate input values

	// UploadPart
	assert.Equal(t, "UPLOAD-ID", val((*args)[1], "UploadId"))
	assert.Equal(t, "UPLOAD-ID", val((*args)[2], "UploadId"))
	assert.Equal(t, "UPLOAD-ID", val((*args)[3], "UploadId"))

	// CompleteMultipartUpload
	assert.Equal(t, "UPLOAD-ID", val((*args)[4], "UploadId"))
	assert.Equal(t, int64(1), val((*args)[4], "MultipartUpload.Parts[0].PartNumber"))
	assert.Equal(t, int64(2), val((*args)[4], "MultipartUpload.Parts[1].PartNumber"))
	assert.Equal(t, int64(3), val((*args)[4], "MultipartUpload.Parts[2].PartNumber"))
	assert.Regexp(t, `^ETAG\d+$`, val((*args)[4], "MultipartUpload.Parts[0].ETag"))
	assert.Regexp(t, `^ETAG\d+$`, val((*args)[4], "MultipartUpload.Parts[1].ETag"))
	assert.Regexp(t, `^ETAG\d+$`, val((*args)[4], "MultipartUpload.Parts[2].ETag"))

	// Custom headers
	assert.Equal(t, "AES256", val((*args)[0], "ServerSideEncryption"))
	assert.Equal(t, "content/type", val((*args)[0], "ContentType"))
}

func TestUploadOrderMultiDifferentPartSize(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.PartSize = 1024 * 1024 * 7
		u.Concurrency = 1
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(buf12MB),
	})

	assert.NoError(t, err)
	assert.Equal(t, []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "CompleteMultipartUpload"}, *ops)

	// Part lengths
	assert.Equal(t, 1024*1024*7, buflen(val((*args)[1], "Body")))
	assert.Equal(t, 1024*1024*5, buflen(val((*args)[2], "Body")))
}

func TestUploadIncreasePartSize(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
		u.MaxUploadParts = 2
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(buf12MB),
	})

	assert.NoError(t, err)
	assert.Equal(t, int64(s3manager.DefaultDownloadPartSize), mgr.PartSize)
	assert.Equal(t, []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "CompleteMultipartUpload"}, *ops)

	// Part lengths
	assert.Equal(t, (1024*1024*6)+1, buflen(val((*args)[1], "Body")))
	assert.Equal(t, (1024*1024*6)-1, buflen(val((*args)[2], "Body")))
}

func TestUploadFailIfPartSizeTooSmall(t *testing.T) {
	mgr := s3manager.NewUploader(unit.Session, func(u *s3manager.Uploader) {
		u.PartSize = 5
	})
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(buf12MB),
	})

	assert.Nil(t, resp)
	assert.NotNil(t, err)

	aerr := err.(awserr.Error)
	assert.Equal(t, "ConfigError", aerr.Code())
	assert.Contains(t, aerr.Message(), "part size must be at least")
}

func TestUploadOrderSingle(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket:               aws.String("Bucket"),
		Key:                  aws.String("Key"),
		Body:                 bytes.NewReader(buf2MB),
		ServerSideEncryption: aws.String("AES256"),
		ContentType:          aws.String("content/type"),
	})

	assert.NoError(t, err)
	assert.Equal(t, []string{"PutObject"}, *ops)
	assert.NotEqual(t, "", resp.Location)
	assert.Equal(t, aws.String("VERSION-ID"), resp.VersionID)
	assert.Equal(t, "", resp.UploadID)
	assert.Equal(t, "AES256", val((*args)[0], "ServerSideEncryption"))
	assert.Equal(t, "content/type", val((*args)[0], "ContentType"))
}

func TestUploadOrderSingleFailure(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	s.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse.StatusCode = 400
	})
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(buf2MB),
	})

	assert.Error(t, err)
	assert.Equal(t, []string{"PutObject"}, *ops)
	assert.Nil(t, resp)
}

func TestUploadOrderZero(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(make([]byte, 0)),
	})

	assert.NoError(t, err)
	assert.Equal(t, []string{"PutObject"}, *ops)
	assert.NotEqual(t, "", resp.Location)
	assert.Equal(t, "", resp.UploadID)
	assert.Equal(t, 0, buflen(val((*args)[0], "Body")))
}

func TestUploadOrderMultiFailure(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	s.Handlers.Send.PushBack(func(r *request.Request) {
		switch t := r.Data.(type) {
		case *s3.UploadPartOutput:
			if *t.ETag == "ETAG2" {
				r.HTTPResponse.StatusCode = 400
			}
		}
	})

	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(buf12MB),
	})

	assert.Error(t, err)
	assert.Equal(t, []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "AbortMultipartUpload"}, *ops)
}

func TestUploadOrderMultiFailureOnComplete(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	s.Handlers.Send.PushBack(func(r *request.Request) {
		switch r.Data.(type) {
		case *s3.CompleteMultipartUploadOutput:
			r.HTTPResponse.StatusCode = 400
		}
	})

	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(buf12MB),
	})

	assert.Error(t, err)
	assert.Equal(t, []string{"CreateMultipartUpload", "UploadPart", "UploadPart",
		"UploadPart", "CompleteMultipartUpload", "AbortMultipartUpload"}, *ops)
}

func TestUploadOrderMultiFailureOnCreate(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	s.Handlers.Send.PushBack(func(r *request.Request) {
		switch r.Data.(type) {
		case *s3.CreateMultipartUploadOutput:
			r.HTTPResponse.StatusCode = 400
		}
	})

	mgr := s3manager.NewUploaderWithClient(s)
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(make([]byte, 1024*1024*12)),
	})

	assert.Error(t, err)
	assert.Equal(t, []string{"CreateMultipartUpload"}, *ops)
}

func TestUploadOrderMultiFailureLeaveParts(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	s.Handlers.Send.PushBack(func(r *request.Request) {
		switch data := r.Data.(type) {
		case *s3.UploadPartOutput:
			if *data.ETag == "ETAG2" {
				r.HTTPResponse.StatusCode = 400
			}
		}
	})

	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
		u.LeavePartsOnError = true
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(make([]byte, 1024*1024*12)),
	})

	assert.Error(t, err)
	assert.Equal(t, []string{"CreateMultipartUpload", "UploadPart", "UploadPart"}, *ops)
}

type failreader struct {
	times     int
	failCount int
}

func (f *failreader) Read(b []byte) (int, error) {
	f.failCount++
	if f.failCount >= f.times {
		return 0, fmt.Errorf("random failure")
	}
	return len(b), nil
}

func TestUploadOrderReadFail1(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &failreader{times: 1},
	})

	assert.Equal(t, "ReadRequestBody", err.(awserr.Error).Code())
	assert.EqualError(t, err.(awserr.Error).OrigErr(), "random failure")
	assert.Equal(t, []string{}, *ops)
}

func TestUploadOrderReadFail2(t *testing.T) {
	s, ops, _ := loggingSvc([]string{"UploadPart"})
	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &failreader{times: 2},
	})

	assert.Equal(t, "ReadRequestBody", err.(awserr.Error).Code())
	assert.EqualError(t, err.(awserr.Error).OrigErr(), "random failure")
	assert.Equal(t, []string{"CreateMultipartUpload", "AbortMultipartUpload"}, *ops)
}

type sizedReader struct {
	size int
	cur  int
}

func (s *sizedReader) Read(p []byte) (n int, err error) {
	if s.cur >= s.size {
		return 0, io.EOF
	}

	n = len(p)
	s.cur += len(p)
	if s.cur > s.size {
		n -= s.cur - s.size
	}

	return
}

func TestUploadOrderMultiBufferedReader(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &sizedReader{size: 1024 * 1024 * 12},
	})

	assert.NoError(t, err)
	assert.Equal(t, []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "UploadPart", "CompleteMultipartUpload"}, *ops)

	// Part lengths
	parts := []int{
		buflen(val((*args)[1], "Body")),
		buflen(val((*args)[2], "Body")),
		buflen(val((*args)[3], "Body")),
	}
	sort.Ints(parts)
	assert.Equal(t, []int{1024 * 1024 * 2, 1024 * 1024 * 5, 1024 * 1024 * 5}, parts)
}

func TestUploadOrderMultiBufferedReaderExceedTotalParts(t *testing.T) {
	s, ops, _ := loggingSvc([]string{"UploadPart"})
	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
		u.MaxUploadParts = 2
	})
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &sizedReader{size: 1024 * 1024 * 12},
	})

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Equal(t, []string{"CreateMultipartUpload", "AbortMultipartUpload"}, *ops)

	aerr := err.(awserr.Error)
	assert.Equal(t, "TotalPartsExceeded", aerr.Code())
	assert.Contains(t, aerr.Message(), "configured MaxUploadParts (2)")
}

func TestUploadOrderSingleBufferedReader(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &sizedReader{size: 1024 * 1024 * 2},
	})

	assert.NoError(t, err)
	assert.Equal(t, []string{"PutObject"}, *ops)
	assert.NotEqual(t, "", resp.Location)
	assert.Equal(t, "", resp.UploadID)
}

func TestUploadZeroLenObject(t *testing.T) {
	requestMade := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestMade = true
		w.WriteHeader(http.StatusOK)
	}))
	mgr := s3manager.NewUploaderWithClient(s3.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	}))
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   strings.NewReader(""),
	})

	assert.NoError(t, err)
	assert.True(t, requestMade)
	assert.NotEqual(t, "", resp.Location)
	assert.Equal(t, "", resp.UploadID)
}
