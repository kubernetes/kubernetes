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

func TestNetworkCreateError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.NetworkCreate(context.Background(), types.NetworkCreate{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestNetworkCreate(t *testing.T) {
	expectedURL := "/networks/create"

	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}

			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}

			content, err := json.Marshal(types.NetworkCreateResponse{
				ID:      "network_id",
				Warning: "warning",
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(content)),
			}, nil
		}),
	}

	networkResponse, err := client.NetworkCreate(context.Background(), types.NetworkCreate{
		Name:           "mynetwork",
		CheckDuplicate: true,
		Driver:         "mydriver",
		EnableIPv6:     true,
		Internal:       true,
		Options: map[string]string{
			"opt-key": "opt-value",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if networkResponse.ID != "network_id" {
		t.Fatalf("expected networkResponse.ID to be 'network_id', got %s", networkResponse.ID)
	}
	if networkResponse.Warning != "warning" {
		t.Fatalf("expected networkResponse.Warning to be 'warning', got %s", networkResponse.Warning)
	}
}
