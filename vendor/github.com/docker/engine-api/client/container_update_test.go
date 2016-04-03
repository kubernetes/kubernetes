package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/engine-api/types/container"
	"golang.org/x/net/context"
)

func TestContainerUpdateError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerUpdate(context.Background(), "nothing", container.UpdateConfig{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerUpdate(t *testing.T) {
	expectedURL := "/containers/container_id/update"
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.ContainerUpdate(context.Background(), "container_id", container.UpdateConfig{
		Resources: container.Resources{
			CPUPeriod: 1,
		},
		RestartPolicy: container.RestartPolicy{
			Name: "always",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}
