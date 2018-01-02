package s3manager_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
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
			start, fin-1, len(data)))
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

func dlLoggingSvcNoContentRangeLength(data []byte, states []int) (*s3.S3, *[]string) {
	var m sync.Mutex
	names := []string{}
	var index int = 0

	svc := s3.New(unit.Session)
	svc.Handlers.Send.Clear()
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		m.Lock()
		defer m.Unlock()

		names = append(names, r.Operation.Name)

		r.HTTPResponse = &http.Response{
			StatusCode: states[index],
			Body:       ioutil.NopCloser(bytes.NewReader(data[:])),
			Header:     http.Header{},
		}
		index++
	})

	return svc, &names
}

func dlLoggingSvcContentRangeTotalAny(data []byte, states []int) (*s3.S3, *[]string) {
	var m sync.Mutex
	names := []string{}
	ranges := []string{}
	var index int = 0

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

		if fin >= int64(len(data)) {
			fin = int64(len(data))
		}

		// Setting start and finish to 0 because this state of 1 is suppose to
		// be an error state of 416
		if index == len(states)-1 {
			start = 0
			fin = 0
		}

		bodyBytes := data[start:fin]

		r.HTTPResponse = &http.Response{
			StatusCode: states[index],
			Body:       ioutil.NopCloser(bytes.NewReader(bodyBytes)),
			Header:     http.Header{},
		}
		r.HTTPResponse.Header.Set("Content-Range", fmt.Sprintf("bytes %d-%d/*",
			start, fin-1))
		index++
	})

	return svc, &names
}

func dlLoggingSvcWithErrReader(cases []testErrReader) (*s3.S3, *[]string) {
	var m sync.Mutex
	names := []string{}
	var index int = 0

	svc := s3.New(unit.Session, &aws.Config{
		MaxRetries: aws.Int(len(cases) - 1),
	})
	svc.Handlers.Send.Clear()
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		m.Lock()
		defer m.Unlock()

		names = append(names, r.Operation.Name)

		c := cases[index]

		r.HTTPResponse = &http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(&c),
			Header:     http.Header{},
		}
		r.HTTPResponse.Header.Set("Content-Range",
			fmt.Sprintf("bytes %d-%d/%d", 0, c.Len-1, c.Len))
		r.HTTPResponse.Header.Set("Content-Length", fmt.Sprintf("%d", c.Len))
		index++
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

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(len(buf12MB)), n; e != a {
		t.Errorf("expect %d buffer length, got %d", e, a)
	}

	expectCalls := []string{"GetObject", "GetObject", "GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}

	expectRngs := []string{"bytes=0-5242879", "bytes=5242880-10485759", "bytes=10485760-15728639"}
	if e, a := expectRngs, *ranges; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v ranges, got %v", e, a)
	}

	count := 0
	for _, b := range w.Bytes() {
		count += int(b)
	}
	if count != 0 {
		t.Errorf("expect 0 count, got %d", count)
	}
}

func TestDownloadZero(t *testing.T) {
	s, names, ranges := dlLoggingSvc([]byte{})

	d := s3manager.NewDownloaderWithClient(s)
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if n != 0 {
		t.Errorf("expect 0 bytes read, got %d", n)
	}
	expectCalls := []string{"GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}

	expectRngs := []string{"bytes=0-5242879"}
	if e, a := expectRngs, *ranges; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v ranges, got %v", e, a)
	}
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

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(3), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject", "GetObject", "GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}
	expectRngs := []string{"bytes=0-0", "bytes=1-1", "bytes=2-2"}
	if e, a := expectRngs, *ranges; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v ranges, got %v", e, a)
	}
	expectBytes := []byte{1, 2, 3}
	if e, a := expectBytes, w.Bytes(); !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v bytes, got %v", e, a)
	}
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

	if err == nil {
		t.Fatalf("expect error, got none")
	}
	aerr := err.(awserr.Error)
	if e, a := "BadRequest", aerr.Code(); e != a {
		t.Errorf("expect %s error code, got %s", e, a)
	}
	if e, a := int64(1), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject", "GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}
	expectBytes := []byte{1}
	if e, a := expectBytes, w.Bytes(); !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v bytes, got %v", e, a)
	}
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

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(len(buf2MB)), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}

	count := 0
	for _, b := range w.Bytes() {
		count += int(b)
	}
	if count != 0 {
		t.Errorf("expect 0 count, got %d", count)
	}
}

func TestDownloadNoContentRangeLength(t *testing.T) {
	s, names := dlLoggingSvcNoContentRangeLength(buf2MB, []int{200, 416})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
	})
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(len(buf2MB)), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject", "GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}

	count := 0
	for _, b := range w.Bytes() {
		count += int(b)
	}
	if count != 0 {
		t.Errorf("expect 0 count, got %d", count)
	}
}

func TestDownloadContentRangeTotalAny(t *testing.T) {
	s, names := dlLoggingSvcContentRangeTotalAny(buf2MB, []int{200, 416})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
	})
	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(len(buf2MB)), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject", "GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}

	count := 0
	for _, b := range w.Bytes() {
		count += int(b)
	}
	if count != 0 {
		t.Errorf("expect 0 count, got %d", count)
	}
}

func TestDownloadPartBodyRetry_SuccessRetry(t *testing.T) {
	s, names := dlLoggingSvcWithErrReader([]testErrReader{
		{Buf: []byte("ab"), Len: 3, Err: io.ErrUnexpectedEOF},
		{Buf: []byte("123"), Len: 3, Err: io.EOF},
	})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
	})

	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(3), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject", "GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}
	if e, a := "123", string(w.Bytes()); e != a {
		t.Errorf("expect %q response, got %q", e, a)
	}
}

func TestDownloadPartBodyRetry_SuccessNoRetry(t *testing.T) {
	s, names := dlLoggingSvcWithErrReader([]testErrReader{
		{Buf: []byte("abc"), Len: 3, Err: io.EOF},
	})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
	})

	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(3), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}
	if e, a := "abc", string(w.Bytes()); e != a {
		t.Errorf("expect %q response, got %q", e, a)
	}
}

func TestDownloadPartBodyRetry_FailRetry(t *testing.T) {
	s, names := dlLoggingSvcWithErrReader([]testErrReader{
		{Buf: []byte("ab"), Len: 3, Err: io.ErrUnexpectedEOF},
	})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 1
	})

	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	if err == nil {
		t.Fatalf("expect error, got none")
	}
	if e, a := "unexpected EOF", err.Error(); !strings.Contains(a, e) {
		t.Errorf("expect %q error message to be in %q", e, a)
	}
	if e, a := int64(2), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}
	if e, a := "ab", string(w.Bytes()); e != a {
		t.Errorf("expect %q response, got %q", e, a)
	}
}

func TestDownloadWithContextCanceled(t *testing.T) {
	d := s3manager.NewDownloader(unit.Session)

	params := s3.GetObjectInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
	}

	ctx := &awstesting.FakeContext{DoneCh: make(chan struct{})}
	ctx.Error = fmt.Errorf("context canceled")
	close(ctx.DoneCh)

	w := &aws.WriteAtBuffer{}

	_, err := d.DownloadWithContext(ctx, w, &params)
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

func TestDownload_WithRange(t *testing.T) {
	s, names, ranges := dlLoggingSvc([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

	d := s3manager.NewDownloaderWithClient(s, func(d *s3manager.Downloader) {
		d.Concurrency = 10 // should be ignored
		d.PartSize = 1     // should be ignored
	})

	w := &aws.WriteAtBuffer{}
	n, err := d.Download(w, &s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
		Range:  aws.String("bytes=2-6"),
	})

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := int64(5), n; e != a {
		t.Errorf("expect %d bytes read, got %d", e, a)
	}
	expectCalls := []string{"GetObject"}
	if e, a := expectCalls, *names; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v API calls, got %v", e, a)
	}
	expectRngs := []string{"bytes=2-6"}
	if e, a := expectRngs, *ranges; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v ranges, got %v", e, a)
	}
	expectBytes := []byte{2, 3, 4, 5, 6}
	if e, a := expectBytes, w.Bytes(); !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v bytes, got %v", e, a)
	}
}

func TestDownload_WithFailure(t *testing.T) {
	svc := s3.New(unit.Session)
	svc.Handlers.Send.Clear()

	first := true
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		if first {
			first = false
			body := bytes.NewReader(make([]byte, s3manager.DefaultDownloadPartSize))
			r.HTTPResponse = &http.Response{
				StatusCode:    http.StatusOK,
				Status:        http.StatusText(http.StatusOK),
				ContentLength: int64(body.Len()),
				Body:          ioutil.NopCloser(body),
				Header:        http.Header{},
			}
			r.HTTPResponse.Header.Set("Content-Length", strconv.Itoa(body.Len()))
			r.HTTPResponse.Header.Set("Content-Range",
				fmt.Sprintf("bytes 0-%d/%d", body.Len()-1, body.Len()*10))
			return
		}

		// Give a chance for the multipart chunks to be queued up
		time.Sleep(1 * time.Second)

		r.HTTPResponse = &http.Response{
			Header: http.Header{},
			Body:   ioutil.NopCloser(&bytes.Buffer{}),
		}
		r.Error = awserr.New("ConnectionError", "some connection error", nil)
		r.Retryable = aws.Bool(false)
	})

	start := time.Now()
	d := s3manager.NewDownloaderWithClient(svc, func(d *s3manager.Downloader) {
		d.Concurrency = 2
	})

	w := &aws.WriteAtBuffer{}
	params := s3.GetObjectInput{
		Bucket: aws.String("Bucket"),
		Key:    aws.String("Key"),
	}

	// Expect this request to exit quickly after failure
	_, err := d.Download(w, &params)
	if err == nil {
		t.Fatalf("expect error, got none")
	}

	limit := start.Add(5 * time.Second)
	dur := time.Now().Sub(start)
	if time.Now().After(limit) {
		t.Errorf("expect time to be less than %v, took %v", limit, dur)
	}
}

type testErrReader struct {
	Buf []byte
	Err error
	Len int64

	off int
}

func (r *testErrReader) Read(p []byte) (int, error) {
	to := len(r.Buf) - r.off

	n := copy(p, r.Buf[r.off:to])
	r.off += n

	if n < len(p) {
		return n, r.Err

	}

	return n, nil
}
