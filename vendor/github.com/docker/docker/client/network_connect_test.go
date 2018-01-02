package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/network"
)

func TestNetworkConnectError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.NetworkConnect(context.Background(), "network_id", "container_id", nil)
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestNetworkConnectEmptyNilEndpointSettings(t *testing.T) {
	expectedURL := "/networks/network_id/connect"

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}

			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}

			var connect types.NetworkConnect
			if err := json.NewDecoder(req.Body).Decode(&connect); err != nil {
				return nil, err
			}

			if connect.Container != "container_id" {
				return nil, fmt.Errorf("expected 'container_id', got %s", connect.Container)
			}

			if connect.EndpointConfig != nil {
				return nil, fmt.Errorf("expected connect.EndpointConfig to be nil, got %v", connect.EndpointConfig)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.NetworkConnect(context.Background(), "network_id", "container_id", nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestNetworkConnect(t *testing.T) {
	expectedURL := "/networks/network_id/connect"

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}

			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}

			var connect types.NetworkConnect
			if err := json.NewDecoder(req.Body).Decode(&connect); err != nil {
				return nil, err
			}

			if connect.Container != "container_id" {
				return nil, fmt.Errorf("expected 'container_id', got %s", connect.Container)
			}

			if connect.EndpointConfig == nil {
				return nil, fmt.Errorf("expected connect.EndpointConfig to be not nil, got %v", connect.EndpointConfig)
			}

			if connect.EndpointConfig.NetworkID != "NetworkID" {
				return nil, fmt.Errorf("expected 'NetworkID', got %s", connect.EndpointConfig.NetworkID)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.NetworkConnect(context.Background(), "network_id", "container_id", &network.EndpointSettings{
		NetworkID: "NetworkID",
	})
	if err != nil {
		t.Fatal(err)
	}
}
