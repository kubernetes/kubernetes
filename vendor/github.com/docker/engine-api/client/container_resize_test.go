package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func TestContainerResizeError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerResize(context.Background(), types.ResizeOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExecResizeError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerExecResize(context.Background(), types.ResizeOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerResize(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, resizeTransport("/containers/container_id/resize")),
	}

	err := client.ContainerResize(context.Background(), types.ResizeOptions{
		ID:     "container_id",
		Height: 500,
		Width:  600,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestContainerExecResize(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, resizeTransport("/exec/exec_id/resize")),
	}

	err := client.ContainerExecResize(context.Background(), types.ResizeOptions{
		ID:     "exec_id",
		Height: 500,
		Width:  600,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func resizeTransport(expectedURL string) func(req *http.Request) (*http.Response, error) {
	return func(req *http.Request) (*http.Response, error) {
		if !strings.HasPrefix(req.URL.Path, expectedURL) {
			return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
		}
		query := req.URL.Query()
		h := query.Get("h")
		if h != "500" {
			return nil, fmt.Errorf("h not set in URL query properly. Expected '500', got %s", h)
		}
		w := query.Get("w")
		if w != "600" {
			return nil, fmt.Errorf("w not set in URL query properly. Expected '600', got %s", w)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
		}, nil
	}
}
