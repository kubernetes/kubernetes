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

func TestDiskUsageError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.DiskUsage(context.Background())
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestDiskUsage(t *testing.T) {
	expectedURL := "/system/df"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}

			du := types.DiskUsage{
				LayersSize: int64(100),
				Images:     nil,
				Containers: nil,
				Volumes:    nil,
			}

			b, err := json.Marshal(du)
			if err != nil {
				return nil, err
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}),
	}
	if _, err := client.DiskUsage(context.Background()); err != nil {
		t.Fatal(err)
	}
}
