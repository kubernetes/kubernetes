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

func TestSecretRemoveUnsupported(t *testing.T) {
	client := &Client{
		version: "1.24",
		client:  &http.Client{},
	}
	err := client.SecretRemove(context.Background(), "secret_id")
	assert.EqualError(t, err, `"secret remove" requires API version 1.25, but the Docker daemon API version is 1.24`)
}

func TestSecretRemoveError(t *testing.T) {
	client := &Client{
		version: "1.25",
		client:  newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.SecretRemove(context.Background(), "secret_id")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestSecretRemove(t *testing.T) {
	expectedURL := "/v1.25/secrets/secret_id"

	client := &Client{
		version: "1.25",
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

	err := client.SecretRemove(context.Background(), "secret_id")
	if err != nil {
		t.Fatal(err)
	}
}
