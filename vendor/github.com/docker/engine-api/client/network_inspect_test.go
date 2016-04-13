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

func TestNetworkInspectError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.NetworkInspect(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestNetworkInspectContainerNotFound(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusNotFound, "Server error")),
	}

	_, err := client.NetworkInspect(context.Background(), "unknown")
	if err == nil || !IsErrNetworkNotFound(err) {
		t.Fatalf("expected a containerNotFound error, got %v", err)
	}
}

func TestNetworkInspect(t *testing.T) {
	expectedURL := "/networks/network_id"
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "GET" {
				return nil, fmt.Errorf("expected GET method, got %s", req.Method)
			}

			content, err := json.Marshal(types.NetworkResource{
				Name: "mynetwork",
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

	r, err := client.NetworkInspect(context.Background(), "network_id")
	if err != nil {
		t.Fatal(err)
	}
	if r.Name != "mynetwork" {
		t.Fatalf("expected `mynetwork`, got %s", r.Name)
	}
}
