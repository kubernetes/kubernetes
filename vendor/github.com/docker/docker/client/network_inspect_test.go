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
	"github.com/docker/docker/api/types/network"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestNetworkInspectError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.NetworkInspect(context.Background(), "nothing", types.NetworkInspectOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestNetworkInspectContainerNotFound(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNotFound, "Server error")),
	}

	_, err := client.NetworkInspect(context.Background(), "unknown", types.NetworkInspectOptions{})
	if err == nil || !IsErrNetworkNotFound(err) {
		t.Fatalf("expected a networkNotFound error, got %v", err)
	}
}

func TestNetworkInspect(t *testing.T) {
	expectedURL := "/networks/network_id"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "GET" {
				return nil, fmt.Errorf("expected GET method, got %s", req.Method)
			}

			var (
				content []byte
				err     error
			)
			if strings.Contains(req.URL.RawQuery, "scope=global") {
				return &http.Response{
					StatusCode: http.StatusNotFound,
					Body:       ioutil.NopCloser(bytes.NewReader(content)),
				}, nil
			}

			if strings.Contains(req.URL.RawQuery, "verbose=true") {
				s := map[string]network.ServiceInfo{
					"web": {},
				}
				content, err = json.Marshal(types.NetworkResource{
					Name:     "mynetwork",
					Services: s,
				})
			} else {
				content, err = json.Marshal(types.NetworkResource{
					Name: "mynetwork",
				})
			}
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(content)),
			}, nil
		}),
	}

	r, err := client.NetworkInspect(context.Background(), "network_id", types.NetworkInspectOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Name != "mynetwork" {
		t.Fatalf("expected `mynetwork`, got %s", r.Name)
	}

	r, err = client.NetworkInspect(context.Background(), "network_id", types.NetworkInspectOptions{Verbose: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.Name != "mynetwork" {
		t.Fatalf("expected `mynetwork`, got %s", r.Name)
	}
	_, ok := r.Services["web"]
	if !ok {
		t.Fatalf("expected service `web` missing in the verbose output")
	}

	_, err = client.NetworkInspect(context.Background(), "network_id", types.NetworkInspectOptions{Scope: "global"})
	assert.EqualError(t, err, "Error: No such network: network_id")
}
