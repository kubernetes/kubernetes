package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/docker/api/types/container"
	"golang.org/x/net/context"
)

func TestContainerCreateError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerCreate(context.Background(), nil, nil, nil, "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error while testing StatusInternalServerError, got %v", err)
	}

	// 404 doesn't automatically means an unknown image
	client = &Client{
		client: newMockClient(errorMock(http.StatusNotFound, "Server error")),
	}
	_, err = client.ContainerCreate(context.Background(), nil, nil, nil, "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error while testing StatusNotFound, got %v", err)
	}
}

func TestContainerCreateImageNotFound(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNotFound, "No such image")),
	}
	_, err := client.ContainerCreate(context.Background(), &container.Config{Image: "unknown_image"}, nil, nil, "unknown")
	if err == nil || !IsErrImageNotFound(err) {
		t.Fatalf("expected an imageNotFound error, got %v", err)
	}
}

func TestContainerCreateWithName(t *testing.T) {
	expectedURL := "/containers/create"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			name := req.URL.Query().Get("name")
			if name != "container_name" {
				return nil, fmt.Errorf("container name not set in URL query properly. Expected `container_name`, got %s", name)
			}
			b, err := json.Marshal(container.ContainerCreateCreatedBody{
				ID: "container_id",
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}),
	}

	r, err := client.ContainerCreate(context.Background(), nil, nil, nil, "container_name")
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "container_id" {
		t.Fatalf("expected `container_id`, got %s", r.ID)
	}
}

// TestContainerCreateAutoRemove validates that a client using API 1.24 always disables AutoRemove. When using API 1.25
// or up, AutoRemove should not be disabled.
func TestContainerCreateAutoRemove(t *testing.T) {
	autoRemoveValidator := func(expectedValue bool) func(req *http.Request) (*http.Response, error) {
		return func(req *http.Request) (*http.Response, error) {
			var config configWrapper

			if err := json.NewDecoder(req.Body).Decode(&config); err != nil {
				return nil, err
			}
			if config.HostConfig.AutoRemove != expectedValue {
				return nil, fmt.Errorf("expected AutoRemove to be %v, got %v", expectedValue, config.HostConfig.AutoRemove)
			}
			b, err := json.Marshal(container.ContainerCreateCreatedBody{
				ID: "container_id",
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}
	}

	client := &Client{
		client:  newMockClient(autoRemoveValidator(false)),
		version: "1.24",
	}
	if _, err := client.ContainerCreate(context.Background(), nil, &container.HostConfig{AutoRemove: true}, nil, ""); err != nil {
		t.Fatal(err)
	}
	client = &Client{
		client:  newMockClient(autoRemoveValidator(true)),
		version: "1.25",
	}
	if _, err := client.ContainerCreate(context.Background(), nil, &container.HostConfig{AutoRemove: true}, nil, ""); err != nil {
		t.Fatal(err)
	}
}
