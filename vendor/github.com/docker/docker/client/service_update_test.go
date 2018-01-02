package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/swarm"
)

func TestServiceUpdateError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.ServiceUpdate(context.Background(), "service_id", swarm.Version{}, swarm.ServiceSpec{}, types.ServiceUpdateOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestServiceUpdate(t *testing.T) {
	expectedURL := "/services/service_id/update"

	updateCases := []struct {
		swarmVersion    swarm.Version
		expectedVersion string
	}{
		{
			expectedVersion: "0",
		},
		{
			swarmVersion: swarm.Version{
				Index: 0,
			},
			expectedVersion: "0",
		},
		{
			swarmVersion: swarm.Version{
				Index: 10,
			},
			expectedVersion: "10",
		},
	}

	for _, updateCase := range updateCases {
		client := &Client{
			client: newMockClient(func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				if req.Method != "POST" {
					return nil, fmt.Errorf("expected POST method, got %s", req.Method)
				}
				version := req.URL.Query().Get("version")
				if version != updateCase.expectedVersion {
					return nil, fmt.Errorf("version not set in URL query properly, expected '%s', got %s", updateCase.expectedVersion, version)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
				}, nil
			}),
		}

		_, err := client.ServiceUpdate(context.Background(), "service_id", updateCase.swarmVersion, swarm.ServiceSpec{}, types.ServiceUpdateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}
}
