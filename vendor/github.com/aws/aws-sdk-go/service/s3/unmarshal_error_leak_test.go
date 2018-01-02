package s3

import (
	"net/http"
	"testing"

	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
)

func TestUnmarhsalErrorLeak(t *testing.T) {
	req := &request.Request{
		HTTPRequest: &http.Request{
			Header: make(http.Header),
			Body:   &awstesting.ReadCloser{Size: 2048},
		},
	}
	req.HTTPResponse = &http.Response{
		Body: &awstesting.ReadCloser{Size: 2048},
		Header: http.Header{
			"X-Amzn-Requestid": []string{"1"},
		},
		StatusCode: http.StatusOK,
	}

	reader := req.HTTPResponse.Body.(*awstesting.ReadCloser)
	unmarshalError(req)

	if req.Error == nil {
		t.Error("expected an error, but received none")
	}

	if !reader.Closed {
		t.Error("expected reader to be closed")
	}

	if e, a := 0, reader.Size; e != a {
		t.Errorf("expected %d, but received %d", e, a)
	}
}
