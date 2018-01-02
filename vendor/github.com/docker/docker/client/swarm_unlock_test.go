package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types/swarm"
)

func TestSwarmUnlockError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.SwarmUnlock(context.Background(), swarm.UnlockRequest{"SWMKEY-1-y6guTZNTwpQeTL5RhUfOsdBdXoQjiB2GADHSRJvbXeU"})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestSwarmUnlock(t *testing.T) {
	expectedURL := "/swarm/unlock"

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.SwarmUnlock(context.Background(), swarm.UnlockRequest{"SWMKEY-1-y6guTZNTwpQeTL5RhUfOsdBdXoQjiB2GADHSRJvbXeU"})
	if err != nil {
		t.Fatal(err)
	}
}
