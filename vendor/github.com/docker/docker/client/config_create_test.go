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

func TestConfigCreateUnsupported(t *testing.T) {
	client := &Client{
		version: "1.29",
		client:  &http.Client{},
	}
	_, err := client.ConfigCreate(context.Background(), swarm.ConfigSpec{})
	assert.EqualError(t, err, `"config create" requires API version 1.30, but the Docker daemon API version is 1.29`)
}

func TestConfigCreateError(t *testing.T) {
	client := &Client{
		version: "1.30",
		client:  newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ConfigCreate(context.Background(), swarm.ConfigSpec{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestConfigCreate(t *testing.T) {
	expectedURL := "/v1.30/configs/create"
	client := &Client{
		version: "1.30",
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}
			b, err := json.Marshal(types.ConfigCreateResponse{
				ID: "test_config",
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

	r, err := client.ConfigCreate(context.Background(), swarm.ConfigSpec{})
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "test_config" {
		t.Fatalf("expected `test_config`, got %s", r.ID)
	}
}
