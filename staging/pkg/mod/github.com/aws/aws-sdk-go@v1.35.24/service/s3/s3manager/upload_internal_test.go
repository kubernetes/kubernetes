// +build go1.7

package s3manager

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	random "math/rand"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/internal/sdkio"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/internal/s3testing"
)

const respBody = `<?xml version="1.0" encoding="UTF-8"?>
<CompleteMultipartUploadOutput>
   <Location>mockValue</Location>
   <Bucket>mockValue</Bucket>
   <Key>mockValue</Key>
   <ETag>mockValue</ETag>
</CompleteMultipartUploadOutput>`

type testReader struct {
	br *bytes.Reader
	m  sync.Mutex
}

func (r *testReader) Read(p []byte) (n int, err error) {
	r.m.Lock()
	defer r.m.Unlock()
	return r.br.Read(p)
}

func TestUploadByteSlicePool(t *testing.T) {
	cases := map[string]struct {
		PartSize      int64
		FileSize      int64
		Concurrency   int
		ExAllocations uint64
	}{
		"single part, single concurrency": {
			PartSize:      sdkio.MebiByte * 5,
			FileSize:      sdkio.MebiByte * 5,
			ExAllocations: 2,
			Concurrency:   1,
		},
		"multi-part, single concurrency": {
			PartSize:      sdkio.MebiByte * 5,
			FileSize:      sdkio.MebiByte * 10,
			ExAllocations: 2,
			Concurrency:   1,
		},
		"multi-part, multiple concurrency": {
			PartSize:      sdkio.MebiByte * 5,
			FileSize:      sdkio.MebiByte * 20,
			ExAllocations: 3,
			Concurrency:   2,
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			var p *recordedPartPool

			unswap := swapByteSlicePool(func(sliceSize int64) byteSlicePool {
				p = newRecordedPartPool(sliceSize)
				return p
			})
			defer unswap()

			sess := unit.Session.Copy()
			svc := s3.New(sess)
			svc.Handlers.Unmarshal.Clear()
			svc.Handlers.UnmarshalMeta.Clear()
			svc.Handlers.UnmarshalError.Clear()
			svc.Handlers.Send.Clear()
			svc.Handlers.Send.PushFront(func(r *request.Request) {
				if r.Body != nil {
					io.Copy(ioutil.Discard, r.Body)
				}

				r.HTTPResponse = &http.Response{
					StatusCode: 200,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(respBody))),
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

			uploader := NewUploaderWithClient(svc, func(u *Uploader) {
				u.PartSize = tt.PartSize
				u.Concurrency = tt.Concurrency
			})

			expected := s3testing.GetTestBytes(int(tt.FileSize))
			_, err := uploader.Upload(&UploadInput{
				Bucket: aws.String("bucket"),
				Key:    aws.String("key"),
				Body:   &testReader{br: bytes.NewReader(expected)},
			})
			if err != nil {
				t.Errorf("expected no error, but got %v", err)
			}

			if v := atomic.LoadInt64(&p.recordedOutstanding); v != 0 {
				t.Fatalf("expected zero outsnatding pool parts, got %d", v)
			}

			gets, allocs := atomic.LoadUint64(&p.recordedGets), atomic.LoadUint64(&p.recordedAllocs)

			t.Logf("total gets %v, total allocations %v", gets, allocs)
			if e, a := tt.ExAllocations, allocs; a > e {
				t.Errorf("expected %v allocations, got %v", e, a)
			}
		})
	}
}

func TestUploadByteSlicePool_Failures(t *testing.T) {
	cases := map[string]struct {
		PartSize   int64
		FileSize   int64
		Operations []string
	}{
		"single part": {
			PartSize: sdkio.MebiByte * 5,
			FileSize: sdkio.MebiByte * 4,
			Operations: []string{
				"PutObject",
			},
		},
		"multi-part": {
			PartSize: sdkio.MebiByte * 5,
			FileSize: sdkio.MebiByte * 10,
			Operations: []string{
				"CreateMultipartUpload",
				"UploadPart",
				"CompleteMultipartUpload",
			},
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			for _, operation := range tt.Operations {
				t.Run(operation, func(t *testing.T) {
					var p *recordedPartPool

					unswap := swapByteSlicePool(func(sliceSize int64) byteSlicePool {
						p = newRecordedPartPool(sliceSize)
						return p
					})
					defer unswap()

					sess := unit.Session.Copy()
					svc := s3.New(sess)
					svc.Handlers.Unmarshal.Clear()
					svc.Handlers.UnmarshalMeta.Clear()
					svc.Handlers.UnmarshalError.Clear()
					svc.Handlers.Send.Clear()
					svc.Handlers.Send.PushFront(func(r *request.Request) {
						if r.Body != nil {
							io.Copy(ioutil.Discard, r.Body)
						}

						if r.Operation.Name == operation {
							r.Retryable = aws.Bool(false)
							r.Error = fmt.Errorf("request error")
							r.HTTPResponse = &http.Response{
								StatusCode: 500,
								Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
							}
							return
						}

						r.HTTPResponse = &http.Response{
							StatusCode: 200,
							Body:       ioutil.NopCloser(bytes.NewReader([]byte(respBody))),
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

					uploader := NewUploaderWithClient(svc, func(u *Uploader) {
						u.Concurrency = 1
						u.PartSize = tt.PartSize
					})

					expected := s3testing.GetTestBytes(int(tt.FileSize))
					_, err := uploader.Upload(&UploadInput{
						Bucket: aws.String("bucket"),
						Key:    aws.String("key"),
						Body:   &testReader{br: bytes.NewReader(expected)},
					})
					if err == nil {
						t.Fatalf("expected error but got none")
					}

					if v := atomic.LoadInt64(&p.recordedOutstanding); v != 0 {
						t.Fatalf("expected zero outsnatding pool parts, got %d", v)
					}
				})
			}
		})
	}
}

func TestUploadByteSlicePoolConcurrentMultiPartSize(t *testing.T) {
	var (
		pools []*recordedPartPool
		mtx   sync.Mutex
	)

	unswap := swapByteSlicePool(func(sliceSize int64) byteSlicePool {
		mtx.Lock()
		defer mtx.Unlock()
		b := newRecordedPartPool(sliceSize)
		pools = append(pools, b)
		return b
	})
	defer unswap()

	sess := unit.Session.Copy()
	svc := s3.New(sess)
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.UnmarshalError.Clear()
	svc.Handlers.Send.Clear()
	svc.Handlers.Send.PushFront(func(r *request.Request) {
		if r.Body != nil {
			io.Copy(ioutil.Discard, r.Body)
		}

		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte(respBody))),
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

	uploader := NewUploaderWithClient(svc, func(u *Uploader) {
		u.PartSize = 5 * sdkio.MebiByte
		u.Concurrency = 2
	})

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			expected := s3testing.GetTestBytes(int(15 * sdkio.MebiByte))
			_, err := uploader.Upload(&UploadInput{
				Bucket: aws.String("bucket"),
				Key:    aws.String("key"),
				Body:   &testReader{br: bytes.NewReader(expected)},
			})
			if err != nil {
				t.Errorf("expected no error, but got %v", err)
			}
		}()
		go func() {
			defer wg.Done()
			expected := s3testing.GetTestBytes(int(15 * sdkio.MebiByte))
			_, err := uploader.Upload(&UploadInput{
				Bucket: aws.String("bucket"),
				Key:    aws.String("key"),
				Body:   &testReader{br: bytes.NewReader(expected)},
			}, func(u *Uploader) {
				u.PartSize = 6 * sdkio.MebiByte
			})
			if err != nil {
				t.Errorf("expected no error, but got %v", err)
			}
		}()
	}

	wg.Wait()

	if e, a := 3, len(pools); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for _, p := range pools {
		if v := atomic.LoadInt64(&p.recordedOutstanding); v != 0 {
			t.Fatalf("expected zero outsnatding pool parts, got %d", v)
		}

		t.Logf("total gets %v, total allocations %v",
			atomic.LoadUint64(&p.recordedGets),
			atomic.LoadUint64(&p.recordedAllocs))
	}
}

func BenchmarkPools(b *testing.B) {
	cases := []struct {
		PartSize      int64
		FileSize      int64
		Concurrency   int
		ExAllocations uint64
	}{
		0: {
			PartSize:    sdkio.MebiByte * 5,
			FileSize:    sdkio.MebiByte * 5,
			Concurrency: 1,
		},
		1: {
			PartSize:    sdkio.MebiByte * 5,
			FileSize:    sdkio.MebiByte * 10,
			Concurrency: 1,
		},
		2: {
			PartSize:    sdkio.MebiByte * 5,
			FileSize:    sdkio.MebiByte * 20,
			Concurrency: 2,
		},
		3: {
			PartSize:    sdkio.MebiByte * 5,
			FileSize:    sdkio.MebiByte * 250,
			Concurrency: 10,
		},
	}

	sess := unit.Session.Copy()
	svc := s3.New(sess)
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.UnmarshalError.Clear()
	svc.Handlers.Send.Clear()
	svc.Handlers.Send.PushFront(func(r *request.Request) {
		if r.Body != nil {
			io.Copy(ioutil.Discard, r.Body)
		}

		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
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

	pools := map[string]func(sliceSize int64) byteSlicePool{
		"sync.Pool": func(sliceSize int64) byteSlicePool {
			return newSyncSlicePool(sliceSize)
		},
		"custom": func(sliceSize int64) byteSlicePool {
			return newMaxSlicePool(sliceSize)
		},
	}

	for name, poolFunc := range pools {
		b.Run(name, func(b *testing.B) {
			unswap := swapByteSlicePool(poolFunc)
			defer unswap()
			for i, c := range cases {
				b.Run(strconv.Itoa(i), func(b *testing.B) {
					uploader := NewUploaderWithClient(svc, func(u *Uploader) {
						u.PartSize = c.PartSize
						u.Concurrency = c.Concurrency
					})

					expected := s3testing.GetTestBytes(int(c.FileSize))
					b.ResetTimer()
					_, err := uploader.Upload(&UploadInput{
						Bucket: aws.String("bucket"),
						Key:    aws.String("key"),
						Body:   &testReader{br: bytes.NewReader(expected)},
					})
					if err != nil {
						b.Fatalf("expected no error, but got %v", err)
					}
				})
			}
		})
	}
}
