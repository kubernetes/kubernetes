package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func TestNetworkDisconnectError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.NetworkDisconnect(context.Background(), "network_id", "container_id", false)
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestNetworkDisconnect(t *testing.T) {
	expectedURL := "/networks/network_id/disconnect"

	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}

			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}

			var disconnect types.NetworkDisconnect
			if err := json.NewDecoder(req.Body).Decode(&disconnect); err != nil {
				return nil, err
			}

			if disconnect.Container != "container_id" {
				return nil, fmt.Errorf("expected 'container_id', got %s", disconnect.Container)
			}

			if !disconnect.Force {
				return nil, fmt.Errorf("expected Force to be true, got %v", disconnect.Force)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.NetworkDisconnect(context.Background(), "network_id", "container_id", true)
	if err != nil {
		t.Fatal(err)
	}
}
