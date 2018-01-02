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

func TestContainerExportError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerExport(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExport(t *testing.T) {
	expectedURL := "/containers/container_id/export"
	client := &Client{
		client: newMockClient(func(r *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(r.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, r.URL)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte("response"))),
			}, nil
		}),
	}
	body, err := client.ContainerExport(context.Background(), "container_id")
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()
	content, err := ioutil.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "response" {
		t.Fatalf("expected response to contain 'response', got %s", string(content))
	}
}
