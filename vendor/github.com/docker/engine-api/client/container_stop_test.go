package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"

	"golang.org/x/net/context"
)

func TestContainerStopError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerStop(context.Background(), "nothing", 0)
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerStop(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
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

	err := client.ContainerStop(context.Background(), "container_id", 100)
	if err != nil {
		t.Fatal(err)
	}
}
