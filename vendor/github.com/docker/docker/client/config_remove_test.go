package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestConfigRemoveUnsupported(t *testing.T) {
	client := &Client{
		version: "1.29",
		client:  &http.Client{},
	}
	err := client.ConfigRemove(context.Background(), "config_id")
	assert.EqualError(t, err, `"config remove" requires API version 1.30, but the Docker daemon API version is 1.29`)
}

func TestConfigRemoveError(t *testing.T) {
	client := &Client{
		version: "1.30",
		client:  newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.ConfigRemove(context.Background(), "config_id")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestConfigRemove(t *testing.T) {
	expectedURL := "/v1.30/configs/config_id"

	client := &Client{
		version: "1.30",
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "DELETE" {
				return nil, fmt.Errorf("expected DELETE method, got %s", req.Method)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte("body"))),
			}, nil
		}),
	}

	err := client.ConfigRemove(context.Background(), "config_id")
	if err != nil {
		t.Fatal(err)
	}
}
