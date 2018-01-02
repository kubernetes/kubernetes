package s3manager

import (
	"bytes"
	"errors"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3iface"
)

func TestHasParity(t *testing.T) {
	cases := []struct {
		o1       *s3.DeleteObjectsInput
		o2       BatchDeleteObject
		expected bool
	}{
		{
			&s3.DeleteObjectsInput{},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{},
			},
			true,
		},
		{
			&s3.DeleteObjectsInput{
				Bucket: aws.String("foo"),
			},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{
					Bucket: aws.String("bar"),
				},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{
					Bucket: aws.String("foo"),
				},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{
				Bucket: aws.String("foo"),
			},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{
				MFA: aws.String("foo"),
			},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{
					MFA: aws.String("bar"),
				},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{
					MFA: aws.String("foo"),
				},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{
				MFA: aws.String("foo"),
			},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{
				RequestPayer: aws.String("foo"),
			},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{
					RequestPayer: aws.String("bar"),
				},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{
					RequestPayer: aws.String("foo"),
				},
			},
			false,
		},
		{
			&s3.DeleteObjectsInput{
				RequestPayer: aws.String("foo"),
			},
			BatchDeleteObject{
				Object: &s3.DeleteObjectInput{},
			},
			false,
		},
	}

	for i, c := range cases {
		if result := hasParity(c.o1, c.o2); result != c.expected {
			t.Errorf("Case %d: expected %t, but received %t\n", i, c.expected, result)
		}
	}
}

func TestBatchDelete(t *testing.T) {
	cases := []struct {
		objects  []BatchDeleteObject
		size     int
		expected int
	}{
		{
			[]BatchDeleteObject{
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("1"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("2"),
						Bucket: aws.String("bucket2"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("3"),
						Bucket: aws.String("bucket3"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("4"),
						Bucket: aws.String("bucket4"),
					},
				},
			},
			1,
			4,
		},
		{
			[]BatchDeleteObject{
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("1"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("2"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("3"),
						Bucket: aws.String("bucket3"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("4"),
						Bucket: aws.String("bucket3"),
					},
				},
			},
			1,
			4,
		},
		{
			[]BatchDeleteObject{
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("1"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("2"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("3"),
						Bucket: aws.String("bucket3"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("4"),
						Bucket: aws.String("bucket3"),
					},
				},
			},
			4,
			2,
		},
		{
			[]BatchDeleteObject{
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("1"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("2"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("3"),
						Bucket: aws.String("bucket3"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("4"),
						Bucket: aws.String("bucket3"),
					},
				},
			},
			10,
			2,
		},
		{
			[]BatchDeleteObject{
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("1"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("2"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("3"),
						Bucket: aws.String("bucket1"),
					},
				},
				{
					Object: &s3.DeleteObjectInput{
						Key:    aws.String("4"),
						Bucket: aws.String("bucket3"),
					},
				},
			},
			2,
			3,
		},
	}

	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
		count++
	}))

	svc := &mockS3Client{S3: buildS3SvcClient(server.URL)}
	for i, c := range cases {
		batcher := BatchDelete{
			Client:    svc,
			BatchSize: c.size,
		}

		if err := batcher.Delete(aws.BackgroundContext(), &DeleteObjectsIterator{Objects: c.objects}); err != nil {
			panic(err)
		}

		if count != c.expected {
			t.Errorf("Case %d: expected %d, but received %d", i, c.expected, count)
		}

		count = 0
	}
}

type mockS3Client struct {
	*s3.S3
	index   int
	objects []*s3.ListObjectsOutput
}

func (client *mockS3Client) ListObjects(input *s3.ListObjectsInput) (*s3.ListObjectsOutput, error) {
	object := client.objects[client.index]
	client.index++
	return object, nil
}

func TestBatchDeleteList(t *testing.T) {
	count := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
		count++
	}))

	objects := []*s3.ListObjectsOutput{
		{
			Contents: []*s3.Object{
				{
					Key: aws.String("1"),
				},
			},
			NextMarker:  aws.String("marker"),
			IsTruncated: aws.Bool(true),
		},
		{
			Contents: []*s3.Object{
				{
					Key: aws.String("2"),
				},
			},
			NextMarker:  aws.String("marker"),
			IsTruncated: aws.Bool(true),
		},
		{
			Contents: []*s3.Object{
				{
					Key: aws.String("3"),
				},
			},
			IsTruncated: aws.Bool(false),
		},
	}

	svc := &mockS3Client{S3: buildS3SvcClient(server.URL), objects: objects}
	batcher := BatchDelete{
		Client:    svc,
		BatchSize: 1,
	}

	input := &s3.ListObjectsInput{
		Bucket: aws.String("bucket"),
	}
	iter := &DeleteListIterator{
		Bucket: input.Bucket,
		Paginator: request.Pagination{
			NewRequest: func() (*request.Request, error) {
				var inCpy *s3.ListObjectsInput
				if input != nil {
					tmp := *input
					inCpy = &tmp
				}
				req, _ := svc.ListObjectsRequest(inCpy)
				req.Handlers.Clear()
				output, _ := svc.ListObjects(inCpy)
				req.Data = output
				return req, nil
			},
		},
	}

	if err := batcher.Delete(aws.BackgroundContext(), iter); err != nil {
		t.Error(err)
	}

	if count != len(objects) {
		t.Errorf("Expected %d, but received %d", len(objects), count)
	}
}

func buildS3SvcClient(u string) *s3.S3 {
	return s3.New(unit.Session, &aws.Config{
		Endpoint:         aws.String(u),
		S3ForcePathStyle: aws.Bool(true),
		DisableSSL:       aws.Bool(true),
		Credentials:      credentials.NewStaticCredentials("AKID", "SECRET", "SESSION"),
	})

}

func TestBatchDeleteList_EmptyListObjects(t *testing.T) {
	count := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
		count++
	}))

	svc := &mockS3Client{S3: buildS3SvcClient(server.URL)}
	batcher := BatchDelete{
		Client: svc,
	}

	input := &s3.ListObjectsInput{
		Bucket: aws.String("bucket"),
	}

	// Test DeleteListIterator in the case when the ListObjectsRequest responds
	// with an empty listing.

	// We need a new iterator with a fresh Pagination since
	// Pagination.HasNextPage() is always true the first time Pagination.Next()
	// called on it
	iter := &DeleteListIterator{
		Bucket: input.Bucket,
		Paginator: request.Pagination{
			NewRequest: func() (*request.Request, error) {
				req, _ := svc.ListObjectsRequest(input)
				// Simulate empty listing
				req.Data = &s3.ListObjectsOutput{Contents: []*s3.Object{}}
				return req, nil
			},
		},
	}

	if err := batcher.Delete(aws.BackgroundContext(), iter); err != nil {
		t.Error(err)
	}
	if count != 1 {
		t.Errorf("expect count to be 1, got %d", count)
	}
}

func TestBatchDownload(t *testing.T) {
	count := 0
	expected := []struct {
		bucket, key string
	}{
		{
			key:    "1",
			bucket: "bucket1",
		},
		{
			key:    "2",
			bucket: "bucket2",
		},
		{
			key:    "3",
			bucket: "bucket3",
		},
		{
			key:    "4",
			bucket: "bucket4",
		},
	}

	received := []struct {
		bucket, key string
	}{}

	payload := []string{
		"1",
		"2",
		"3",
		"4",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		urlParts := strings.Split(r.URL.String(), "/")
		received = append(received, struct{ bucket, key string }{urlParts[1], urlParts[2]})
		w.Write([]byte(payload[count]))
		count++
	}))

	svc := NewDownloaderWithClient(buildS3SvcClient(server.URL))

	objects := []BatchDownloadObject{
		{
			Object: &s3.GetObjectInput{
				Key:    aws.String("1"),
				Bucket: aws.String("bucket1"),
			},
			Writer: aws.NewWriteAtBuffer(make([]byte, 128)),
		},
		{
			Object: &s3.GetObjectInput{
				Key:    aws.String("2"),
				Bucket: aws.String("bucket2"),
			},
			Writer: aws.NewWriteAtBuffer(make([]byte, 128)),
		},
		{
			Object: &s3.GetObjectInput{
				Key:    aws.String("3"),
				Bucket: aws.String("bucket3"),
			},
			Writer: aws.NewWriteAtBuffer(make([]byte, 128)),
		},
		{
			Object: &s3.GetObjectInput{
				Key:    aws.String("4"),
				Bucket: aws.String("bucket4"),
			},
			Writer: aws.NewWriteAtBuffer(make([]byte, 128)),
		},
	}

	iter := &DownloadObjectsIterator{Objects: objects}
	if err := svc.DownloadWithIterator(aws.BackgroundContext(), iter); err != nil {
		panic(err)
	}

	if count != len(objects) {
		t.Errorf("Expected %d, but received %d", len(objects), count)
	}

	if len(expected) != len(received) {
		t.Errorf("Expected %d, but received %d", len(expected), len(received))
	}

	for i := 0; i < len(expected); i++ {
		if expected[i].key != received[i].key {
			t.Errorf("Expected %q, but received %q", expected[i].key, received[i].key)
		}

		if expected[i].bucket != received[i].bucket {
			t.Errorf("Expected %q, but received %q", expected[i].bucket, received[i].bucket)
		}
	}

	for i, p := range payload {
		b := iter.Objects[i].Writer.(*aws.WriteAtBuffer).Bytes()
		b = bytes.Trim(b, "\x00")

		if string(b) != p {
			t.Errorf("Expected %q, but received %q", p, b)
		}
	}
}

func TestBatchUpload(t *testing.T) {
	count := 0
	expected := []struct {
		bucket, key string
		reqBody     string
	}{
		{
			key:     "1",
			bucket:  "bucket1",
			reqBody: "1",
		},
		{
			key:     "2",
			bucket:  "bucket2",
			reqBody: "2",
		},
		{
			key:     "3",
			bucket:  "bucket3",
			reqBody: "3",
		},
		{
			key:     "4",
			bucket:  "bucket4",
			reqBody: "4",
		},
	}

	received := []struct {
		bucket, key, reqBody string
	}{}

	payload := []string{
		"a",
		"b",
		"c",
		"d",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		urlParts := strings.Split(r.URL.String(), "/")

		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}

		received = append(received, struct{ bucket, key, reqBody string }{urlParts[1], urlParts[2], string(b)})
		w.Write([]byte(payload[count]))

		count++
	}))

	svc := NewUploaderWithClient(buildS3SvcClient(server.URL))

	objects := []BatchUploadObject{
		{
			Object: &UploadInput{
				Key:    aws.String("1"),
				Bucket: aws.String("bucket1"),
				Body:   bytes.NewBuffer([]byte("1")),
			},
		},
		{
			Object: &UploadInput{
				Key:    aws.String("2"),
				Bucket: aws.String("bucket2"),
				Body:   bytes.NewBuffer([]byte("2")),
			},
		},
		{
			Object: &UploadInput{
				Key:    aws.String("3"),
				Bucket: aws.String("bucket3"),
				Body:   bytes.NewBuffer([]byte("3")),
			},
		},
		{
			Object: &UploadInput{
				Key:    aws.String("4"),
				Bucket: aws.String("bucket4"),
				Body:   bytes.NewBuffer([]byte("4")),
			},
		},
	}

	iter := &UploadObjectsIterator{Objects: objects}
	if err := svc.UploadWithIterator(aws.BackgroundContext(), iter); err != nil {
		panic(err)
	}

	if count != len(objects) {
		t.Errorf("Expected %d, but received %d", len(objects), count)
	}

	if len(expected) != len(received) {
		t.Errorf("Expected %d, but received %d", len(expected), len(received))
	}

	for i := 0; i < len(expected); i++ {
		if expected[i].key != received[i].key {
			t.Errorf("Expected %q, but received %q", expected[i].key, received[i].key)
		}

		if expected[i].bucket != received[i].bucket {
			t.Errorf("Expected %q, but received %q", expected[i].bucket, received[i].bucket)
		}

		if expected[i].reqBody != received[i].reqBody {
			t.Errorf("Expected %q, but received %q", expected[i].reqBody, received[i].reqBody)
		}
	}
}

type mockClient struct {
	s3iface.S3API
	Put       func() (*s3.PutObjectOutput, error)
	Get       func() (*s3.GetObjectOutput, error)
	List      func() (*s3.ListObjectsOutput, error)
	responses []response
}

type response struct {
	out interface{}
	err error
}

func (client *mockClient) PutObject(input *s3.PutObjectInput) (*s3.PutObjectOutput, error) {
	return client.Put()
}

func (client *mockClient) PutObjectRequest(input *s3.PutObjectInput) (*request.Request, *s3.PutObjectOutput) {
	req, _ := client.S3API.PutObjectRequest(input)
	req.Handlers.Clear()
	req.Data, req.Error = client.Put()
	return req, req.Data.(*s3.PutObjectOutput)
}

func (client *mockClient) ListObjects(input *s3.ListObjectsInput) (*s3.ListObjectsOutput, error) {
	return client.List()
}

func (client *mockClient) ListObjectsRequest(input *s3.ListObjectsInput) (*request.Request, *s3.ListObjectsOutput) {
	req, _ := client.S3API.ListObjectsRequest(input)
	req.Handlers.Clear()
	req.Data, req.Error = client.List()
	return req, req.Data.(*s3.ListObjectsOutput)
}

func TestBatchError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	}))

	index := 0
	responses := []response{
		{
			&s3.PutObjectOutput{},
			errors.New("Foo"),
		},
		{
			&s3.PutObjectOutput{},
			nil,
		},
		{
			&s3.PutObjectOutput{},
			nil,
		},
		{
			&s3.PutObjectOutput{},
			errors.New("Bar"),
		},
	}

	svc := &mockClient{
		S3API: buildS3SvcClient(server.URL),
		Put: func() (*s3.PutObjectOutput, error) {
			resp := responses[index]
			index++
			return resp.out.(*s3.PutObjectOutput), resp.err
		},
		List: func() (*s3.ListObjectsOutput, error) {
			resp := responses[index]
			index++
			return resp.out.(*s3.ListObjectsOutput), resp.err
		},
	}
	uploader := NewUploaderWithClient(svc)

	objects := []BatchUploadObject{
		{
			Object: &UploadInput{
				Key:    aws.String("1"),
				Bucket: aws.String("bucket1"),
				Body:   bytes.NewBuffer([]byte("1")),
			},
		},
		{
			Object: &UploadInput{
				Key:    aws.String("2"),
				Bucket: aws.String("bucket2"),
				Body:   bytes.NewBuffer([]byte("2")),
			},
		},
		{
			Object: &UploadInput{
				Key:    aws.String("3"),
				Bucket: aws.String("bucket3"),
				Body:   bytes.NewBuffer([]byte("3")),
			},
		},
		{
			Object: &UploadInput{
				Key:    aws.String("4"),
				Bucket: aws.String("bucket4"),
				Body:   bytes.NewBuffer([]byte("4")),
			},
		},
	}

	iter := &UploadObjectsIterator{Objects: objects}
	if err := uploader.UploadWithIterator(aws.BackgroundContext(), iter); err != nil {
		if bErr, ok := err.(*BatchError); !ok {
			t.Error("Expected BatchError, but received other")
		} else {
			if len(bErr.Errors) != 2 {
				t.Errorf("Expected 2 errors, but received %d", len(bErr.Errors))
			}

			expected := []struct {
				bucket, key string
			}{
				{
					"bucket1",
					"1",
				},
				{
					"bucket4",
					"4",
				},
			}
			for i, expect := range expected {
				if *bErr.Errors[i].Bucket != expect.bucket {
					t.Errorf("Case %d: Invalid bucket expected %s, but received %s", i, expect.bucket, *bErr.Errors[i].Bucket)
				}

				if *bErr.Errors[i].Key != expect.key {
					t.Errorf("Case %d: Invalid key expected %s, but received %s", i, expect.key, *bErr.Errors[i].Key)
				}
			}
		}
	} else {
		t.Error("Expected error, but received nil")
	}

	if index != len(objects) {
		t.Errorf("Expected %d, but received %d", len(objects), index)
	}

}

type testAfterDeleteIter struct {
	afterDelete   bool
	afterDownload bool
	afterUpload   bool
	next          bool
}

func (iter *testAfterDeleteIter) Next() bool {
	next := !iter.next
	iter.next = !iter.next
	return next
}

func (iter *testAfterDeleteIter) Err() error {
	return nil
}

func (iter *testAfterDeleteIter) DeleteObject() BatchDeleteObject {
	return BatchDeleteObject{
		Object: &s3.DeleteObjectInput{
			Bucket: aws.String("foo"),
			Key:    aws.String("foo"),
		},
		After: func() error {
			iter.afterDelete = true
			return nil
		},
	}
}

type testAfterDownloadIter struct {
	afterDownload bool
	afterUpload   bool
	next          bool
}

func (iter *testAfterDownloadIter) Next() bool {
	next := !iter.next
	iter.next = !iter.next
	return next
}

func (iter *testAfterDownloadIter) Err() error {
	return nil
}

func (iter *testAfterDownloadIter) DownloadObject() BatchDownloadObject {
	return BatchDownloadObject{
		Object: &s3.GetObjectInput{
			Bucket: aws.String("foo"),
			Key:    aws.String("foo"),
		},
		Writer: aws.NewWriteAtBuffer([]byte{}),
		After: func() error {
			iter.afterDownload = true
			return nil
		},
	}
}

type testAfterUploadIter struct {
	afterUpload bool
	next        bool
}

func (iter *testAfterUploadIter) Next() bool {
	next := !iter.next
	iter.next = !iter.next
	return next
}

func (iter *testAfterUploadIter) Err() error {
	return nil
}

func (iter *testAfterUploadIter) UploadObject() BatchUploadObject {
	return BatchUploadObject{
		Object: &UploadInput{
			Bucket: aws.String("foo"),
			Key:    aws.String("foo"),
			Body:   strings.NewReader("bar"),
		},
		After: func() error {
			iter.afterUpload = true
			return nil
		},
	}
}

func TestAfter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	}))

	index := 0
	responses := []response{
		{
			&s3.PutObjectOutput{},
			nil,
		},
		{
			&s3.GetObjectOutput{},
			nil,
		},
		{
			&s3.DeleteObjectOutput{},
			nil,
		},
	}

	svc := &mockClient{
		S3API: buildS3SvcClient(server.URL),
		Put: func() (*s3.PutObjectOutput, error) {
			resp := responses[index]
			index++
			return resp.out.(*s3.PutObjectOutput), resp.err
		},
		Get: func() (*s3.GetObjectOutput, error) {
			resp := responses[index]
			index++
			return resp.out.(*s3.GetObjectOutput), resp.err
		},
		List: func() (*s3.ListObjectsOutput, error) {
			resp := responses[index]
			index++
			return resp.out.(*s3.ListObjectsOutput), resp.err
		},
	}
	uploader := NewUploaderWithClient(svc)
	downloader := NewDownloaderWithClient(svc)
	deleter := NewBatchDeleteWithClient(svc)

	deleteIter := &testAfterDeleteIter{}
	downloadIter := &testAfterDownloadIter{}
	uploadIter := &testAfterUploadIter{}

	if err := uploader.UploadWithIterator(aws.BackgroundContext(), uploadIter); err != nil {
		t.Error(err)
	}

	if err := downloader.DownloadWithIterator(aws.BackgroundContext(), downloadIter); err != nil {
		t.Error(err)
	}

	if err := deleter.Delete(aws.BackgroundContext(), deleteIter); err != nil {
		t.Error(err)
	}

	if !deleteIter.afterDelete {
		t.Error("Expected 'afterDelete' to be true, but received false")
	}

	if !downloadIter.afterDownload {
		t.Error("Expected 'afterDownload' to be true, but received false")
	}

	if !uploadIter.afterUpload {
		t.Error("Expected 'afterUpload' to be true, but received false")
	}
}
