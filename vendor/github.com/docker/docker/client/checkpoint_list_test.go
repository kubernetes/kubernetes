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
	"golang.org/x/net/context"
)

func TestCheckpointListError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.CheckpointList(context.Background(), "container_id", types.CheckpointListOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestCheckpointList(t *testing.T) {
	expectedURL := "/containers/container_id/checkpoints"

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			content, err := json.Marshal([]types.Checkpoint{
				{
					Name: "checkpoint",
				},
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(content)),
			}, nil
		}),
	}

	checkpoints, err := client.CheckpointList(context.Background(), "container_id", types.CheckpointListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(checkpoints) != 1 {
		t.Fatalf("expected 1 checkpoint, got %v", checkpoints)
	}
}

func TestCheckpointListContainerNotFound(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNotFound, "Server error")),
	}

	_, err := client.CheckpointList(context.Background(), "unknown", types.CheckpointListOptions{})
	if err == nil || !IsErrContainerNotFound(err) {
		t.Fatalf("expected a containerNotFound error, got %v", err)
	}
}
