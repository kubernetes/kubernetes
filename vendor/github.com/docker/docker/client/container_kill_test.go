package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"
)

func TestContainerKillError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerKill(context.Background(), "nothing", "SIGKILL")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerKill(t *testing.T) {
	expectedURL := "/containers/container_id/kill"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			signal := req.URL.Query().Get("signal")
			if signal != "SIGKILL" {
				return nil, fmt.Errorf("signal not set in URL query properly. Expected 'SIGKILL', got %s", signal)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.ContainerKill(context.Background(), "container_id", "SIGKILL")
	if err != nil {
		t.Fatal(err)
	}
}
