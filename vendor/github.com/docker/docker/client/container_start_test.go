package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
)

func TestContainerStartError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerStart(context.Background(), "nothing", types.ContainerStartOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerStart(t *testing.T) {
	expectedURL := "/containers/container_id/start"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			// we're not expecting any payload, but if one is supplied, check it is valid.
			if req.Header.Get("Content-Type") == "application/json" {
				var startConfig interface{}
				if err := json.NewDecoder(req.Body).Decode(&startConfig); err != nil {
					return nil, fmt.Errorf("Unable to parse json: %s", err)
				}
			}

			checkpoint := req.URL.Query().Get("checkpoint")
			if checkpoint != "checkpoint_id" {
				return nil, fmt.Errorf("checkpoint not set in URL query properly. Expected 'checkpoint_id', got %s", checkpoint)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.ContainerStart(context.Background(), "container_id", types.ContainerStartOptions{CheckpointID: "checkpoint_id"})
	if err != nil {
		t.Fatal(err)
	}
}
