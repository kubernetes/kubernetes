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

func TestSwarmInitError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.SwarmInit(context.Background(), swarm.InitRequest{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestSwarmInit(t *testing.T) {
	expectedURL := "/swarm/init"

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
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(`"body"`))),
			}, nil
		}),
	}

	resp, err := client.SwarmInit(context.Background(), swarm.InitRequest{
		ListenAddr: "0.0.0.0:2377",
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp != "body" {
		t.Fatalf("Expected 'body', got %s", resp)
	}
}
