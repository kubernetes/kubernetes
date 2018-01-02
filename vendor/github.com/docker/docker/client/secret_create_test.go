package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/swarm"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestSecretCreateUnsupported(t *testing.T) {
	client := &Client{
		version: "1.24",
		client:  &http.Client{},
	}
	_, err := client.SecretCreate(context.Background(), swarm.SecretSpec{})
	assert.EqualError(t, err, `"secret create" requires API version 1.25, but the Docker daemon API version is 1.24`)
}

func TestSecretCreateError(t *testing.T) {
	client := &Client{
		version: "1.25",
		client:  newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.SecretCreate(context.Background(), swarm.SecretSpec{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestSecretCreate(t *testing.T) {
	expectedURL := "/v1.25/secrets/create"
	client := &Client{
		version: "1.25",
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}
			b, err := json.Marshal(types.SecretCreateResponse{
				ID: "test_secret",
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusCreated,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}),
	}

	r, err := client.SecretCreate(context.Background(), swarm.SecretSpec{})
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "test_secret" {
		t.Fatalf("expected `test_secret`, got %s", r.ID)
	}
}
