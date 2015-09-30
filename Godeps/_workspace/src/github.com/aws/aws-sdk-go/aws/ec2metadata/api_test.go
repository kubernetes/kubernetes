package ec2metadata_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
)

func initTestServer(path string, resp string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.RequestURI != path {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		w.Write([]byte(resp))
	}))
}

func TestEndpoint(t *testing.T) {
	c := ec2metadata.New(&ec2metadata.Config{})
	op := &request.Operation{
		Name:       "GetMetadata",
		HTTPMethod: "GET",
		HTTPPath:   path.Join("/", "meta-data", "testpath"),
	}

	req := c.Service.NewRequest(op, nil, nil)
	assert.Equal(t, "http://169.254.169.254/latest", req.Service.Endpoint)
	assert.Equal(t, "http://169.254.169.254/latest/meta-data/testpath", req.HTTPRequest.URL.String())
}

func TestGetMetadata(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/some/path",
		"success", // real response includes suffix
	)
	defer server.Close()
	c := ec2metadata.New(&ec2metadata.Config{Endpoint: aws.String(server.URL + "/latest")})

	resp, err := c.GetMetadata("some/path")

	assert.NoError(t, err)
	assert.Equal(t, "success", resp)
}

func TestGetRegion(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/placement/availability-zone",
		"us-west-2a", // real response includes suffix
	)
	defer server.Close()
	c := ec2metadata.New(&ec2metadata.Config{Endpoint: aws.String(server.URL + "/latest")})

	region, err := c.Region()

	assert.NoError(t, err)
	assert.Equal(t, "us-west-2", region)
}

func TestMetadataAvailable(t *testing.T) {
	server := initTestServer(
		"/latest/meta-data/instance-id",
		"instance-id",
	)
	defer server.Close()
	c := ec2metadata.New(&ec2metadata.Config{Endpoint: aws.String(server.URL + "/latest")})

	available := c.Available()

	assert.True(t, available)
}

func TestMetadataNotAvailable(t *testing.T) {
	c := ec2metadata.New(nil)
	c.Handlers.Send.Clear()
	c.Handlers.Send.PushBack(func(r *request.Request) {
		r.HTTPResponse = &http.Response{
			StatusCode: int(0),
			Status:     http.StatusText(int(0)),
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
		}
		r.Error = awserr.New("RequestError", "send request failed", nil)
		r.Retryable = aws.Bool(true) // network errors are retryable
	})

	available := c.Available()

	assert.False(t, available)
}
