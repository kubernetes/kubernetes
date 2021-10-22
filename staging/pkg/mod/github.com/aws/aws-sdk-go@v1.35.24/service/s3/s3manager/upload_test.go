// +build go1.8

package s3manager_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	random "math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/internal/s3testing"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

var emptyList = []string{}

const respMsg = `<?xml version="1.0" encoding="UTF-8"?>
<CompleteMultipartUploadOutput>
   <Location>mockValue</Location>
   <Bucket>mockValue</Bucket>
   <Key>mockValue</Key>
   <ETag>mockValue</ETag>
</CompleteMultipartUploadOutput>`

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
			Body:       ioutil.NopCloser(bytes.NewReader([]byte(respMsg))),
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
		Key:                  aws.String("Key - value"),
		Body:                 bytes.NewReader(buf12MB),
		ServerSideEncryption: aws.String("aws:kms"),
		SSEKMSKeyId:          aws.String("KmsId"),
		ContentType:          aws.String("content/type"),
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	expected := []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "UploadPart", "CompleteMultipartUpload"}
	if !reflect.DeepEqual(expected, *ops) {
		t.Errorf("Expected %v, but received %v", expected, *ops)
	}

	if e, a := `https://s3.mock-region.amazonaws.com/Bucket/Key%20-%20value`, resp.Location; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if "UPLOAD-ID" != resp.UploadID {
		t.Errorf("Expected %q, but received %q", "UPLOAD-ID", resp.UploadID)
	}

	if "VERSION-ID" != *resp.VersionID {
		t.Errorf("Expected %q, but received %q", "VERSION-ID", *resp.VersionID)
	}

	// Validate input values

	// UploadPart
	for i := 1; i < 5; i++ {
		v := val((*args)[i], "UploadId")
		if "UPLOAD-ID" != v {
			t.Errorf("Expected %q, but received %q", "UPLOAD-ID", v)
		}
	}

	// CompleteMultipartUpload
	v := val((*args)[4], "UploadId")
	if "UPLOAD-ID" != v {
		t.Errorf("Expected %q, but received %q", "UPLOAD-ID", v)
	}

	for i := 0; i < 3; i++ {
		e := val((*args)[4], fmt.Sprintf("MultipartUpload.Parts[%d].PartNumber", i))
		if int64(i+1) != e.(int64) {
			t.Errorf("Expected %d, but received %d", i+1, e)
		}
	}

	vals := []string{
		val((*args)[4], "MultipartUpload.Parts[0].ETag").(string),
		val((*args)[4], "MultipartUpload.Parts[1].ETag").(string),
		val((*args)[4], "MultipartUpload.Parts[2].ETag").(string),
	}

	for _, a := range vals {
		if matched, err := regexp.MatchString(`^ETAG\d+$`, a); !matched || err != nil {
			t.Errorf("Failed regexp expression `^ETAG\\d+$`")
		}
	}

	// Custom headers
	e := val((*args)[0], "ServerSideEncryption")
	if e != "aws:kms" {
		t.Errorf("Expected %q, but received %q", "aws:kms", e)
	}

	e = val((*args)[0], "SSEKMSKeyId")
	if e != "KmsId" {
		t.Errorf("Expected %q, but received %q", "KmsId", e)
	}

	e = val((*args)[0], "ContentType")
	if e != "content/type" {
		t.Errorf("Expected %q, but received %q", "content/type", e)
	}
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

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	vals := []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "CompleteMultipartUpload"}
	if !reflect.DeepEqual(vals, *ops) {
		t.Errorf("Expected %v, but received %v", vals, *ops)
	}

	// Part lengths
	if len := buflen(val((*args)[1], "Body")); 1024*1024*7 != len {
		t.Errorf("Expected %d, but received %d", 1024*1024*7, len)
	}
	if len := buflen(val((*args)[2], "Body")); 1024*1024*5 != len {
		t.Errorf("Expected %d, but received %d", 1024*1024*5, len)
	}
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

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if int64(s3manager.DefaultDownloadPartSize) != mgr.PartSize {
		t.Errorf("Expected %d, but received %d", s3manager.DefaultDownloadPartSize, mgr.PartSize)
	}

	vals := []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "CompleteMultipartUpload"}
	if !reflect.DeepEqual(vals, *ops) {
		t.Errorf("Expected %v, but received %v", vals, *ops)
	}

	// Part lengths
	if len := buflen(val((*args)[1], "Body")); (1024*1024*6)+1 != len {
		t.Errorf("Expected %d, but received %d", (1024*1024*6)+1, len)
	}

	if len := buflen(val((*args)[2], "Body")); (1024*1024*6)-1 != len {
		t.Errorf("Expected %d, but received %d", (1024*1024*6)-1, len)
	}
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

	if resp != nil {
		t.Errorf("Expected response to be nil, but received %v", resp)
	}

	if err == nil {
		t.Errorf("Expected error, but received nil")
	}

	aerr := err.(awserr.Error)
	if e, a := "ConfigError", aerr.Code(); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := "part size must be at least", aerr.Message(); !strings.Contains(a, e) {
		t.Errorf("expect %v to be in %v", e, a)
	}
}

func TestUploadOrderSingle(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket:               aws.String("Bucket"),
		Key:                  aws.String("Key - value"),
		Body:                 bytes.NewReader(buf2MB),
		ServerSideEncryption: aws.String("aws:kms"),
		SSEKMSKeyId:          aws.String("KmsId"),
		ContentType:          aws.String("content/type"),
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if vals := []string{"PutObject"}; !reflect.DeepEqual(vals, *ops) {
		t.Errorf("Expected %v, but received %v", vals, *ops)
	}

	if e, a := `https://s3.mock-region.amazonaws.com/Bucket/Key%20-%20value`, resp.Location; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e := "VERSION-ID"; e != *resp.VersionID {
		t.Errorf("Expected %q, but received %q", e, *resp.VersionID)
	}

	if len(resp.UploadID) > 0 {
		t.Errorf("Expected empty string, but received %q", resp.UploadID)
	}

	if e, a := "aws:kms", val((*args)[0], "ServerSideEncryption").(string); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := "KmsId", val((*args)[0], "SSEKMSKeyId").(string); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := "content/type", val((*args)[0], "ContentType").(string); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}
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

	if err == nil {
		t.Error("Expected error, but receievd nil")
	}

	if vals := []string{"PutObject"}; !reflect.DeepEqual(vals, *ops) {
		t.Errorf("Expected %v, but received %v", vals, *ops)
	}

	if resp != nil {
		t.Errorf("Expected response to be nil, but received %v", resp)
	}
}

func TestUploadOrderZero(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(make([]byte, 0)),
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if vals := []string{"PutObject"}; !reflect.DeepEqual(vals, *ops) {
		t.Errorf("Expected %v, but received %v", vals, *ops)
	}

	if len(resp.Location) == 0 {
		t.Error("Expected Location to not be empty")
	}

	if len(resp.UploadID) > 0 {
		t.Errorf("Expected empty string, but received %q", resp.UploadID)
	}

	if e, a := 0, buflen(val((*args)[0], "Body")); e != a {
		t.Errorf("Expected %d, but received %d", e, a)
	}
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

	if err == nil {
		t.Error("Expected error, but receievd nil")
	}

	if e, a := []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "AbortMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but received %v", e, a)
	}
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

	if err == nil {
		t.Error("Expected error, but receievd nil")
	}

	if e, a := []string{"CreateMultipartUpload", "UploadPart", "UploadPart",
		"UploadPart", "CompleteMultipartUpload", "AbortMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but received %v", e, a)
	}
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

	if err == nil {
		t.Error("Expected error, but receievd nil")
	}

	if e, a := []string{"CreateMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but received %v", e, a)
	}
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

	if err == nil {
		t.Error("Expected error, but receievd nil")
	}

	if e, a := []string{"CreateMultipartUpload", "UploadPart", "UploadPart"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but received %v", e, a)
	}
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

	if e, a := "ReadRequestBody", err.(awserr.Error).Code(); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := err.(awserr.Error).OrigErr().Error(), "random failure"; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := []string{}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but received %v", e, a)
	}
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

	if e, a := "MultipartUpload", err.(awserr.Error).Code(); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := "ReadRequestBody", err.(awserr.Error).OrigErr().(awserr.Error).Code(); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if errStr := err.(awserr.Error).OrigErr().Error(); !strings.Contains(errStr, "random failure") {
		t.Errorf("Expected error to contains 'random failure', but was %q", errStr)
	}

	if e, a := []string{"CreateMultipartUpload", "AbortMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}
}

type sizedReader struct {
	size int
	cur  int
	err  error
}

func (s *sizedReader) Read(p []byte) (n int, err error) {
	if s.cur >= s.size {
		if s.err == nil {
			s.err = io.EOF
		}
		return 0, s.err
	}

	n = len(p)
	s.cur += len(p)
	if s.cur > s.size {
		n -= s.cur - s.size
	}

	return n, err
}

func TestUploadOrderMultiBufferedReader(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &sizedReader{size: 1024 * 1024 * 12},
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if e, a := []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "UploadPart", "CompleteMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}

	// Part lengths
	parts := []int{
		buflen(val((*args)[1], "Body")),
		buflen(val((*args)[2], "Body")),
		buflen(val((*args)[3], "Body")),
	}
	sort.Ints(parts)

	if e, a := []int{1024 * 1024 * 2, 1024 * 1024 * 5, 1024 * 1024 * 5}, parts; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}
}

func TestUploadOrderMultiBufferedReaderPartial(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &sizedReader{size: 1024 * 1024 * 12, err: io.EOF},
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if e, a := []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "UploadPart", "CompleteMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}

	// Part lengths
	parts := []int{
		buflen(val((*args)[1], "Body")),
		buflen(val((*args)[2], "Body")),
		buflen(val((*args)[3], "Body")),
	}
	sort.Ints(parts)

	if e, a := []int{1024 * 1024 * 2, 1024 * 1024 * 5, 1024 * 1024 * 5}, parts; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}
}

// TestUploadOrderMultiBufferedReaderEOF tests the edge case where the
// file size is the same as part size.
func TestUploadOrderMultiBufferedReaderEOF(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &sizedReader{size: 1024 * 1024 * 10, err: io.EOF},
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if e, a := []string{"CreateMultipartUpload", "UploadPart", "UploadPart", "CompleteMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}

	// Part lengths
	parts := []int{
		buflen(val((*args)[1], "Body")),
		buflen(val((*args)[2], "Body")),
	}
	sort.Ints(parts)

	if e, a := []int{1024 * 1024 * 5, 1024 * 1024 * 5}, parts; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}
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

	if err == nil {
		t.Error("Expected an error, but received nil")
	}

	if resp != nil {
		t.Errorf("Expected nil, but receievd %v", resp)
	}

	if e, a := []string{"CreateMultipartUpload", "AbortMultipartUpload"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but receievd %v", e, a)
	}

	aerr := err.(awserr.Error)
	if e, a := "MultipartUpload", aerr.Code(); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := "TotalPartsExceeded", aerr.OrigErr().(awserr.Error).Code(); e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if !strings.Contains(aerr.Error(), "configured MaxUploadParts (2)") {
		t.Errorf("Expected error to contain 'configured MaxUploadParts (2)', but receievd %q", aerr.Error())
	}
}

func TestUploadOrderSingleBufferedReader(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &sizedReader{size: 1024 * 1024 * 2},
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if e, a := []string{"PutObject"}, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %v, but received %v", e, a)
	}

	if len(resp.Location) == 0 {
		t.Error("Expected a value in Location but received empty string")
	}

	if len(resp.UploadID) > 0 {
		t.Errorf("Expected empty string but received %q", resp.UploadID)
	}
}

func TestUploadZeroLenObject(t *testing.T) {
	requestMade := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestMade = true
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	mgr := s3manager.NewUploaderWithClient(s3.New(unit.Session, &aws.Config{
		Endpoint: aws.String(server.URL),
	}))
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   strings.NewReader(""),
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}
	if !requestMade {
		t.Error("Expected request to have been made, but was not")
	}

	if len(resp.Location) == 0 {
		t.Error("Expected a non-empty string value for Location")
	}

	if len(resp.UploadID) > 0 {
		t.Errorf("Expected empty string, but received %q", resp.UploadID)
	}
}

func TestUploadInputS3PutObjectInputPairity(t *testing.T) {
	matchings := compareStructType(reflect.TypeOf(s3.PutObjectInput{}),
		reflect.TypeOf(s3manager.UploadInput{}))
	aOnly := []string{}
	bOnly := []string{}

	for k, c := range matchings {
		if c == 1 && k != "ContentLength" {
			aOnly = append(aOnly, k)
		} else if c == 2 {
			bOnly = append(bOnly, k)
		}
	}

	if len(aOnly) > 0 {
		t.Errorf("Expected empty array, but received %v", aOnly)
	}

	if len(bOnly) > 0 {
		t.Errorf("Expected empty array, but received %v", bOnly)
	}
}

type testIncompleteReader struct {
	Size int64
	read int64
}

func (r *testIncompleteReader) Read(p []byte) (n int, err error) {
	r.read += int64(len(p))
	if r.read >= r.Size {
		return int(r.read - r.Size), io.ErrUnexpectedEOF
	}
	return len(p), nil
}

func TestUploadUnexpectedEOF(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
		u.PartSize = s3manager.MinUploadPartSize
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body: &testIncompleteReader{
			Size: int64(s3manager.MinUploadPartSize + 1),
		},
	})
	if err == nil {
		t.Error("Expected error, but received none")
	}

	// Ensure upload started.
	if e, a := "CreateMultipartUpload", (*ops)[0]; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	// Part may or may not be sent because of timing of sending parts and
	// reading next part in upload manager. Just check for the last abort.
	if e, a := "AbortMultipartUpload", (*ops)[len(*ops)-1]; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}
}

func compareStructType(a, b reflect.Type) map[string]int {
	if a.Kind() != reflect.Struct || b.Kind() != reflect.Struct {
		panic(fmt.Sprintf("types must both be structs, got %v and %v", a.Kind(), b.Kind()))
	}

	aFields := enumFields(a)
	bFields := enumFields(b)

	matchings := map[string]int{}

	for i := 0; i < len(aFields) || i < len(bFields); i++ {
		if i < len(aFields) {
			c := matchings[aFields[i].Name]
			matchings[aFields[i].Name] = c + 1
		}
		if i < len(bFields) {
			c := matchings[bFields[i].Name]
			matchings[bFields[i].Name] = c + 2
		}
	}

	return matchings
}

func enumFields(v reflect.Type) []reflect.StructField {
	fields := []reflect.StructField{}

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		// Ignoreing anon fields
		if field.PkgPath != "" {
			// Ignore unexported fields
			continue
		}

		fields = append(fields, field)
	}

	return fields
}

type fooReaderAt struct{}

func (r *fooReaderAt) Read(p []byte) (n int, err error) {
	return 12, io.EOF
}

func (r *fooReaderAt) ReadAt(p []byte, off int64) (n int, err error) {
	return 12, io.EOF
}

func TestReaderAt(t *testing.T) {
	svc := s3.New(unit.Session)
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.UnmarshalError.Clear()
	svc.Handlers.Send.Clear()

	contentLen := ""
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		contentLen = r.HTTPRequest.Header.Get("Content-Length")
		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
		}
	})

	mgr := s3manager.NewUploaderWithClient(svc, func(u *s3manager.Uploader) {
		u.Concurrency = 1
	})

	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   &fooReaderAt{},
	})

	if err != nil {
		t.Errorf("Expected no error but received %v", err)
	}

	if e, a := "12", contentLen; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}
}

func TestSSE(t *testing.T) {
	svc := s3.New(unit.Session)
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.UnmarshalError.Clear()
	svc.Handlers.ValidateResponse.Clear()
	svc.Handlers.Send.Clear()
	partNum := 0
	mutex := &sync.Mutex{}

	svc.Handlers.Send.PushBack(func(r *request.Request) {
		mutex.Lock()
		defer mutex.Unlock()
		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte(respMsg))),
		}
		switch data := r.Data.(type) {
		case *s3.CreateMultipartUploadOutput:
			data.UploadId = aws.String("UPLOAD-ID")
		case *s3.UploadPartOutput:
			input := r.Params.(*s3.UploadPartInput)
			if input.SSECustomerAlgorithm == nil {
				t.Fatal("SSECustomerAlgoritm should not be nil")
			}
			if input.SSECustomerKey == nil {
				t.Fatal("SSECustomerKey should not be nil")
			}
			partNum++
			data.ETag = aws.String(fmt.Sprintf("ETAG%d", partNum))
		case *s3.CompleteMultipartUploadOutput:
			data.Location = aws.String("https://location")
			data.VersionId = aws.String("VERSION-ID")
		case *s3.PutObjectOutput:
			data.VersionId = aws.String("VERSION-ID")
		}

	})

	mgr := s3manager.NewUploaderWithClient(svc, func(u *s3manager.Uploader) {
		u.Concurrency = 5
	})

	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket:               aws.String("Bucket"),
		Key:                  aws.String("Key"),
		SSECustomerAlgorithm: aws.String("AES256"),
		SSECustomerKey:       aws.String("foo"),
		Body:                 bytes.NewBuffer(make([]byte, 1024*1024*10)),
	})

	if err != nil {
		t.Fatal("Expected no error, but received" + err.Error())
	}
}

func TestUploadWithContextCanceled(t *testing.T) {
	u := s3manager.NewUploader(unit.Session)

	params := s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   bytes.NewReader(make([]byte, 0)),
	}

	ctx := &awstesting.FakeContext{DoneCh: make(chan struct{})}
	ctx.Error = fmt.Errorf("context canceled")
	close(ctx.DoneCh)

	_, err := u.UploadWithContext(ctx, &params)
	if err == nil {
		t.Fatalf("expected error, did not get one")
	}
	aerr := err.(awserr.Error)
	if e, a := request.CanceledErrorCode, aerr.Code(); e != a {
		t.Errorf("expected error code %q, got %q", e, a)
	}
	if e, a := "canceled", aerr.Message(); !strings.Contains(a, e) {
		t.Errorf("expected error message to contain %q, but did not %q", e, a)
	}
}

// S3 Uploader incorrectly fails an upload if the content being uploaded
// has a size of MinPartSize * MaxUploadParts.
// Github:  aws/aws-sdk-go#2557
func TestUploadMaxPartsEOF(t *testing.T) {
	s, ops, _ := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
		u.PartSize = s3manager.DefaultUploadPartSize
		u.MaxUploadParts = 2
	})
	f := bytes.NewReader(make([]byte, int(mgr.PartSize)*mgr.MaxUploadParts))

	r1 := io.NewSectionReader(f, 0, s3manager.DefaultUploadPartSize)
	r2 := io.NewSectionReader(f, s3manager.DefaultUploadPartSize, 2*s3manager.DefaultUploadPartSize)
	body := io.MultiReader(r1, r2)

	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body:   body,
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expectOps := []string{
		"CreateMultipartUpload",
		"UploadPart",
		"UploadPart",
		"CompleteMultipartUpload",
	}
	if e, a := expectOps, *ops; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v ops, got %v", e, a)
	}
}

func createTempFile(t *testing.T, size int64) (*os.File, func(*testing.T), error) {
	file, err := ioutil.TempFile(os.TempDir(), aws.SDKName+t.Name())
	if err != nil {
		return nil, nil, err
	}
	filename := file.Name()
	if err := file.Truncate(size); err != nil {
		return nil, nil, err
	}

	return file,
		func(t *testing.T) {
			if err := file.Close(); err != nil {
				t.Errorf("failed to close temp file, %s, %v", filename, err)
			}
			if err := os.Remove(filename); err != nil {
				t.Errorf("failed to remove temp file, %s, %v", filename, err)
			}
		},
		nil
}

func buildFailHandlers(tb testing.TB, parts, retry int) []http.Handler {
	handlers := make([]http.Handler, parts)
	for i := 0; i < len(handlers); i++ {
		handlers[i] = &failPartHandler{
			tb:             tb,
			failsRemaining: retry,
			successHandler: successPartHandler{tb: tb},
		}
	}

	return handlers
}

func TestUploadRetry(t *testing.T) {
	const numParts, retries = 3, 10

	testFile, testFileCleanup, err := createTempFile(t, s3manager.DefaultUploadPartSize*numParts)
	if err != nil {
		t.Fatalf("failed to create test file, %v", err)
	}
	defer testFileCleanup(t)

	cases := map[string]struct {
		Body         io.Reader
		PartHandlers func(testing.TB) []http.Handler
	}{
		"bytes.Buffer": {
			Body: bytes.NewBuffer(make([]byte, s3manager.DefaultUploadPartSize*numParts)),
			PartHandlers: func(tb testing.TB) []http.Handler {
				return buildFailHandlers(tb, numParts, retries)
			},
		},
		"bytes.Reader": {
			Body: bytes.NewReader(make([]byte, s3manager.DefaultUploadPartSize*numParts)),
			PartHandlers: func(tb testing.TB) []http.Handler {
				return buildFailHandlers(tb, numParts, retries)
			},
		},
		"os.File": {
			Body: testFile,
			PartHandlers: func(tb testing.TB) []http.Handler {
				return buildFailHandlers(tb, numParts, retries)
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			mux := newMockS3UploadServer(t, c.PartHandlers(t))
			server := httptest.NewServer(mux)
			defer server.Close()

			var logger aws.Logger
			var logLevel *aws.LogLevelType
			if v := os.Getenv("DEBUG_BODY"); len(v) != 0 {
				logger = t
				logLevel = aws.LogLevel(
					aws.LogDebugWithRequestErrors | aws.LogDebugWithRequestRetries,
				)
			}
			sess := unit.Session.Copy(&aws.Config{
				Endpoint:         aws.String(server.URL),
				S3ForcePathStyle: aws.Bool(true),
				DisableSSL:       aws.Bool(true),
				MaxRetries:       aws.Int(retries + 1),
				SleepDelay:       func(time.Duration) {},

				Logger:   logger,
				LogLevel: logLevel,
				//Credentials: credentials.AnonymousCredentials,
			})

			uploader := s3manager.NewUploader(sess, func(u *s3manager.Uploader) {
				//				u.Concurrency = 1
			})
			_, err := uploader.Upload(&s3manager.UploadInput{
				Bucket: aws.String("bucket"),
				Key:    aws.String("key"),
				Body:   c.Body,
			})

			if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}
		})
	}
}

func TestUploadBufferStrategy(t *testing.T) {
	cases := map[string]struct {
		PartSize  int64
		Size      int64
		Strategy  s3manager.ReadSeekerWriteToProvider
		callbacks int
	}{
		"NoBuffer": {
			PartSize: s3manager.DefaultUploadPartSize,
			Strategy: nil,
		},
		"SinglePart": {
			PartSize:  s3manager.DefaultUploadPartSize,
			Size:      s3manager.DefaultUploadPartSize,
			Strategy:  &recordedBufferProvider{size: int(s3manager.DefaultUploadPartSize)},
			callbacks: 1,
		},
		"MultiPart": {
			PartSize:  s3manager.DefaultUploadPartSize,
			Size:      s3manager.DefaultUploadPartSize * 2,
			Strategy:  &recordedBufferProvider{size: int(s3manager.DefaultUploadPartSize)},
			callbacks: 2,
		},
	}

	for name, tCase := range cases {
		t.Run(name, func(t *testing.T) {
			_ = tCase
			sess := unit.Session.Copy()
			svc := s3.New(sess)
			svc.Handlers.Unmarshal.Clear()
			svc.Handlers.UnmarshalMeta.Clear()
			svc.Handlers.UnmarshalError.Clear()
			svc.Handlers.Send.Clear()
			svc.Handlers.Send.PushBack(func(r *request.Request) {
				if r.Body != nil {
					io.Copy(ioutil.Discard, r.Body)
				}

				r.HTTPResponse = &http.Response{
					StatusCode: 200,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(respMsg))),
				}

				switch data := r.Data.(type) {
				case *s3.CreateMultipartUploadOutput:
					data.UploadId = aws.String("UPLOAD-ID")
				case *s3.UploadPartOutput:
					data.ETag = aws.String(fmt.Sprintf("ETAG%d", random.Int()))
				case *s3.CompleteMultipartUploadOutput:
					data.Location = aws.String("https://location")
					data.VersionId = aws.String("VERSION-ID")
				case *s3.PutObjectOutput:
					data.VersionId = aws.String("VERSION-ID")
				}
			})

			uploader := s3manager.NewUploaderWithClient(svc, func(u *s3manager.Uploader) {
				u.PartSize = tCase.PartSize
				u.BufferProvider = tCase.Strategy
				u.Concurrency = 1
			})

			expected := s3testing.GetTestBytes(int(tCase.Size))
			_, err := uploader.Upload(&s3manager.UploadInput{
				Bucket: aws.String("bucket"),
				Key:    aws.String("key"),
				Body:   bytes.NewReader(expected),
			})
			if err != nil {
				t.Fatalf("failed to upload file: %v", err)
			}

			switch strat := tCase.Strategy.(type) {
			case *recordedBufferProvider:
				if !bytes.Equal(expected, strat.content) {
					t.Errorf("content buffered did not match expected")
				}
				if tCase.callbacks != strat.callbackCount {
					t.Errorf("expected %v, got %v callbacks", tCase.callbacks, strat.callbackCount)
				}
			}
		})
	}
}

type mockS3UploadServer struct {
	*http.ServeMux

	tb          testing.TB
	partHandler []http.Handler
}

func newMockS3UploadServer(tb testing.TB, partHandler []http.Handler) *mockS3UploadServer {
	s := &mockS3UploadServer{
		ServeMux:    http.NewServeMux(),
		partHandler: partHandler,
		tb:          tb,
	}

	s.HandleFunc("/", s.handleRequest)

	return s
}

func (s mockS3UploadServer) handleRequest(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	_, hasUploads := r.URL.Query()["uploads"]

	switch {
	case r.Method == "POST" && hasUploads:
		// CreateMultipartUpload
		w.Header().Set("Content-Length", strconv.Itoa(len(createUploadResp)))
		w.Write([]byte(createUploadResp))

	case r.Method == "PUT":
		// UploadPart
		partNumStr := r.URL.Query().Get("partNumber")
		id, err := strconv.Atoi(partNumStr)
		if err != nil {
			failRequest(w, 400, "BadRequest",
				fmt.Sprintf("unable to parse partNumber, %q, %v",
					partNumStr, err))
			return
		}
		id--
		if id < 0 || id >= len(s.partHandler) {
			failRequest(w, 400, "BadRequest",
				fmt.Sprintf("invalid partNumber %v", id))
			return
		}
		s.partHandler[id].ServeHTTP(w, r)

	case r.Method == "POST":
		// CompleteMultipartUpload
		w.Header().Set("Content-Length", strconv.Itoa(len(completeUploadResp)))
		w.Write([]byte(completeUploadResp))

	case r.Method == "DELETE":
		// AbortMultipartUpload
		w.Header().Set("Content-Length", strconv.Itoa(len(abortUploadResp)))
		w.WriteHeader(200)
		w.Write([]byte(abortUploadResp))

	default:
		failRequest(w, 400, "BadRequest",
			fmt.Sprintf("invalid request %v %v", r.Method, r.URL))
	}
}

func failRequest(w http.ResponseWriter, status int, code, msg string) {
	msg = fmt.Sprintf(baseRequestErrorResp, code, msg)
	w.Header().Set("Content-Length", strconv.Itoa(len(msg)))
	w.WriteHeader(status)
	w.Write([]byte(msg))
}

type successPartHandler struct {
	tb testing.TB
}

func (h successPartHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	n, err := io.Copy(ioutil.Discard, r.Body)
	if err != nil {
		failRequest(w, 400, "BadRequest",
			fmt.Sprintf("failed to read body, %v", err))
		return
	}

	contLenStr := r.Header.Get("Content-Length")
	expectLen, err := strconv.ParseInt(contLenStr, 10, 64)
	if err != nil {
		h.tb.Logf("expect content-length, got %q, %v", contLenStr, err)
		failRequest(w, 400, "BadRequest",
			fmt.Sprintf("unable to get content-length %v", err))
		return
	}
	if e, a := expectLen, n; e != a {
		h.tb.Logf("expect %v read, got %v", e, a)
		failRequest(w, 400, "BadRequest",
			fmt.Sprintf(
				"content-length and body do not match, %v, %v", e, a))
		return
	}

	w.Header().Set("Content-Length", strconv.Itoa(len(uploadPartResp)))
	w.Write([]byte(uploadPartResp))
}

type failPartHandler struct {
	tb testing.TB

	failsRemaining int
	successHandler http.Handler
}

func (h *failPartHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	if h.failsRemaining == 0 && h.successHandler != nil {
		h.successHandler.ServeHTTP(w, r)
		return
	}

	io.Copy(ioutil.Discard, r.Body)

	failRequest(w, 500, "InternalException",
		fmt.Sprintf("mock error, partNumber %v", r.URL.Query().Get("partNumber")))

	h.failsRemaining--
}

type recordedBufferProvider struct {
	content       []byte
	size          int
	callbackCount int
}

func (r *recordedBufferProvider) GetWriteTo(seeker io.ReadSeeker) (s3manager.ReadSeekerWriteTo, func()) {
	b := make([]byte, r.size)
	w := &s3manager.BufferedReadSeekerWriteTo{BufferedReadSeeker: s3manager.NewBufferedReadSeeker(seeker, b)}

	return w, func() {
		r.content = append(r.content, b...)
		r.callbackCount++
	}
}

const createUploadResp = `
<CreateMultipartUploadResponse>
  <Bucket>bucket</Bucket>
  <Key>key</Key>
  <UploadId>abc123</UploadId>
</CreateMultipartUploadResponse>
`
const uploadPartResp = `
<UploadPartResponse>
  <ETag>key</ETag>
</UploadPartResponse>
`
const baseRequestErrorResp = `
<Error>
  <Code>%s</Code>
  <Message>%s</Message>
  <RequestId>request-id</RequestId>
  <HostId>host-id</HostId>
</Error>
`
const completeUploadResp = `
<CompleteMultipartUploadResponse>
  <Bucket>bucket</Bucket>
  <Key>key</Key>
  <ETag>key</ETag>
  <Location>https://bucket.us-west-2.amazonaws.com/key</Location>
  <UploadId>abc123</UploadId>
</CompleteMultipartUploadResponse>
`

const abortUploadResp = `
<AbortMultipartUploadResponse>
</AbortMultipartUploadResponse>
`
