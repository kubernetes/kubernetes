package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"

	"golang.org/x/net/context"
)

func TestContainerKillError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerKill(context.Background(), "nothing", "SIGKILL")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerKill(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
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
