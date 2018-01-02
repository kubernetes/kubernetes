package s3manager_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
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

	if "https://location" != resp.Location {
		t.Errorf("Expected %q, but received %q", "https://location", resp.Location)
	}

	if "UPLOAD-ID" != resp.UploadID {
		t.Errorf("Expected %q, but received %q", "UPLOAD-ID", resp.UploadID)
	}

	if "VERSION-ID" != *resp.VersionID {
		t.Errorf("Expected %q, but received %q", "VERSION-ID", resp.VersionID)
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
	if "ConfigError" != aerr.Code() {
		t.Errorf("Expected %q, but received %q", "ConfigError", aerr.Code())
	}

	if strings.Contains("part size must be at least", aerr.Message()) {
		t.Errorf("Expected string to contain %q, but received %q", "part size must be at least", aerr.Message())
	}
}

func TestUploadOrderSingle(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s)
	resp, err := mgr.Upload(&s3manager.UploadInput{
		Bucket:               aws.String("Bucket"),
		Key:                  aws.String("Key"),
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

	if len(resp.Location) == 0 {
		t.Error("Expected Location to not be empty")
	}

	if e := "VERSION-ID"; e != *resp.VersionID {
		t.Errorf("Expected %q, but received %q", e, resp.VersionID)
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
	Buf   []byte
	Count int
}

func (r *testIncompleteReader) Read(p []byte) (n int, err error) {
	if r.Count < 0 {
		return 0, io.ErrUnexpectedEOF
	}

	r.Count--
	return copy(p, r.Buf), nil
}

func TestUploadUnexpectedEOF(t *testing.T) {
	s, ops, args := loggingSvc(emptyList)
	mgr := s3manager.NewUploaderWithClient(s, func(u *s3manager.Uploader) {
		u.Concurrency = 1
	})
	_, err := mgr.Upload(&s3manager.UploadInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
		Body: &testIncompleteReader{
			Buf:   make([]byte, 1024*1024*5),
			Count: 1,
		},
	})

	if err == nil {
		t.Error("Expected error, but received none")
	}

	if e, a := "CreateMultipartUpload", (*ops)[0]; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := "UploadPart", (*ops)[1]; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	if e, a := "AbortMultipartUpload", (*ops)[len(*ops)-1]; e != a {
		t.Errorf("Expected %q, but received %q", e, a)
	}

	// Part lengths
	if e, a := 1024*1024*5, buflen(val((*args)[1], "Body")); e != a {
		t.Errorf("Expected %d, but received %d", e, a)
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
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
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
