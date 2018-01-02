package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestContainerStopError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	timeout := 0 * time.Second
	err := client.ContainerStop(context.Background(), "nothing", &timeout)
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerStop(t *testing.T) {
	expectedURL := "/containers/container_id/stop"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			t := req.URL.Query().Get("t")
			if t != "100" {
				return nil, fmt.Errorf("t (timeout) not set in URL query properly. Expected '100', got %s", t)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}
	timeout := 100 * time.Second
	err := client.ContainerStop(context.Background(), "container_id", &timeout)
	if err != nil {
		t.Fatal(err)
	}
}
