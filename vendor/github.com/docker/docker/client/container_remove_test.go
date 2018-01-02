package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

func TestContainerRemoveError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerRemove(context.Background(), "container_id", types.ContainerRemoveOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerRemove(t *testing.T) {
	expectedURL := "/containers/container_id"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			query := req.URL.Query()
			volume := query.Get("v")
			if volume != "1" {
				return nil, fmt.Errorf("v (volume) not set in URL query properly. Expected '1', got %s", volume)
			}
			force := query.Get("force")
			if force != "1" {
				return nil, fmt.Errorf("force not set in URL query properly. Expected '1', got %s", force)
			}
			link := query.Get("link")
			if link != "" {
				return nil, fmt.Errorf("link should have not be present in query, go %s", link)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.ContainerRemove(context.Background(), "container_id", types.ContainerRemoveOptions{
		RemoveVolumes: true,
		Force:         true,
	})
	if err != nil {
		t.Fatal(err)
	}
}
