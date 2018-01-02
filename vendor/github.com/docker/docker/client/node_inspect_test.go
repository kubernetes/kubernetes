package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/docker/api/types/swarm"
	"golang.org/x/net/context"
)

func TestNodeInspectError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, _, err := client.NodeInspectWithRaw(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestNodeInspectNodeNotFound(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNotFound, "Server error")),
	}

	_, _, err := client.NodeInspectWithRaw(context.Background(), "unknown")
	if err == nil || !IsErrNodeNotFound(err) {
		t.Fatalf("expected a nodeNotFoundError error, got %v", err)
	}
}

func TestNodeInspect(t *testing.T) {
	expectedURL := "/nodes/node_id"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			content, err := json.Marshal(swarm.Node{
				ID: "node_id",
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

	nodeInspect, _, err := client.NodeInspectWithRaw(context.Background(), "node_id")
	if err != nil {
		t.Fatal(err)
	}
	if nodeInspect.ID != "node_id" {
		t.Fatalf("expected `node_id`, got %s", nodeInspect.ID)
	}
}
