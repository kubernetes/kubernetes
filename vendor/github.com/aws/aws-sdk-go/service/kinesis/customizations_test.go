package kinesis

import (
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

type testReader struct {
	duration time.Duration
}

func (r *testReader) Read(b []byte) (int, error) {
	time.Sleep(r.duration)
	return 0, io.EOF
}

func (r *testReader) Close() error {
	return nil
}

// GetRecords will hang unexpectedly during reads.
// See https://github.com/aws/aws-sdk-go/issues/1141
func TestKinesisGetRecordsCustomization(t *testing.T) {
	readDuration = time.Millisecond
	retryCount := 0
	svc := New(unit.Session, &aws.Config{
		MaxRetries: aws.Int(4),
	})
	req, _ := svc.GetRecordsRequest(&GetRecordsInput{
		ShardIterator: aws.String("foo"),
	})
	req.Handlers.Send.Clear()
	req.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Header: http.Header{
				"X-Amz-Request-Id": []string{"abc123"},
			},
			Body:          &testReader{duration: 10 * time.Second},
			ContentLength: -1,
		}
		r.HTTPResponse.Status = http.StatusText(r.HTTPResponse.StatusCode)
		retryCount++
	})
	req.ApplyOptions(request.WithResponseReadTimeout(time.Second))
	err := req.Send()
	if err == nil {
		t.Errorf("Expected error, but received nil")
	} else if v, ok := err.(awserr.Error); !ok {
		t.Errorf("Expected awserr.Error but received %v", err)
	} else if v.Code() != request.ErrCodeResponseTimeout {
		t.Errorf("Expected 'RequestTimeout' error, but received %s instead", v.Code())
	}
	if retryCount != 5 {
		t.Errorf("Expected '5' retries, but received %d", retryCount)
	}
}

func TestKinesisGetRecordsNoTimeout(t *testing.T) {
	readDuration = time.Second
	svc := New(unit.Session)
	req, _ := svc.GetRecordsRequest(&GetRecordsInput{
		ShardIterator: aws.String("foo"),
	})
	req.Handlers.Send.Clear()
	req.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &http.Response{
			StatusCode: 200,
			Header: http.Header{
				"X-Amz-Request-Id": []string{"abc123"},
			},
			Body:          &testReader{duration: time.Duration(0)},
			ContentLength: -1,
		}
		r.HTTPResponse.Status = http.StatusText(r.HTTPResponse.StatusCode)
	})
	req.ApplyOptions(request.WithResponseReadTimeout(time.Second))
	err := req.Send()
	if err != nil {
		t.Errorf("Expected no error, but received %v", err)
	}
}
