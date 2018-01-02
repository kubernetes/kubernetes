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

func TestContainerInspectError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.ContainerInspect(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerInspectContainerNotFound(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNotFound, "Server error")),
	}

	_, err := client.ContainerInspect(context.Background(), "unknown")
	if err == nil || !IsErrContainerNotFound(err) {
		t.Fatalf("expected a containerNotFound error, got %v", err)
	}
}

func TestContainerInspect(t *testing.T) {
	expectedURL := "/containers/container_id/json"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			content, err := json.Marshal(types.ContainerJSON{
				ContainerJSONBase: &types.ContainerJSONBase{
					ID:    "container_id",
					Image: "image",
					Name:  "name",
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

	r, err := client.ContainerInspect(context.Background(), "container_id")
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "container_id" {
		t.Fatalf("expected `container_id`, got %s", r.ID)
	}
	if r.Image != "image" {
		t.Fatalf("expected `image`, got %s", r.Image)
	}
	if r.Name != "name" {
		t.Fatalf("expected `name`, got %s", r.Name)
	}
}

func TestContainerInspectNode(t *testing.T) {
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			content, err := json.Marshal(types.ContainerJSON{
				ContainerJSONBase: &types.ContainerJSONBase{
					ID:    "container_id",
					Image: "image",
					Name:  "name",
					Node: &types.ContainerNode{
						ID:     "container_node_id",
						Addr:   "container_node",
						Labels: map[string]string{"foo": "bar"},
					},
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

	r, err := client.ContainerInspect(context.Background(), "container_id")
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "container_id" {
		t.Fatalf("expected `container_id`, got %s", r.ID)
	}
	if r.Image != "image" {
		t.Fatalf("expected `image`, got %s", r.Image)
	}
	if r.Name != "name" {
		t.Fatalf("expected `name`, got %s", r.Name)
	}
	if r.Node.ID != "container_node_id" {
		t.Fatalf("expected `container_node_id`, got %s", r.Node.ID)
	}
	if r.Node.Addr != "container_node" {
		t.Fatalf("expected `container_node`, got %s", r.Node.Addr)
	}
	foo, ok := r.Node.Labels["foo"]
	if foo != "bar" || !ok {
		t.Fatalf("expected `bar` for label `foo`")
	}
}
