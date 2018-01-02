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

func TestSwarmLeaveError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.SwarmLeave(context.Background(), false)
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestSwarmLeave(t *testing.T) {
	expectedURL := "/swarm/leave"

	leaveCases := []struct {
		force         bool
		expectedForce string
	}{
		{
			expectedForce: "",
		},
		{
			force:         true,
			expectedForce: "1",
		},
	}

	for _, leaveCase := range leaveCases {
		client := &Client{
			client: newMockClient(func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				if req.Method != "POST" {
					return nil, fmt.Errorf("expected POST method, got %s", req.Method)
				}
				force := req.URL.Query().Get("force")
				if force != leaveCase.expectedForce {
					return nil, fmt.Errorf("force not set in URL query properly. expected '%s', got %s", leaveCase.expectedForce, force)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
				}, nil
			}),
		}

		err := client.SwarmLeave(context.Background(), leaveCase.force)
		if err != nil {
			t.Fatal(err)
		}
	}
}
