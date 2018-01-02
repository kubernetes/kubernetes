package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"

	"golang.org/x/net/context"
)

func TestNodeRemoveError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.NodeRemove(context.Background(), "node_id", types.NodeRemoveOptions{Force: false})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestNodeRemove(t *testing.T) {
	expectedURL := "/nodes/node_id"

	removeCases := []struct {
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

	for _, removeCase := range removeCases {
		client := &Client{
			client: newMockClient(func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				if req.Method != "DELETE" {
					return nil, fmt.Errorf("expected DELETE method, got %s", req.Method)
				}
				force := req.URL.Query().Get("force")
				if force != removeCase.expectedForce {
					return nil, fmt.Errorf("force not set in URL query properly. expected '%s', got %s", removeCase.expectedForce, force)
				}

				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte("body"))),
				}, nil
			}),
		}

		err := client.NodeRemove(context.Background(), "node_id", types.NodeRemoveOptions{Force: removeCase.force})
		if err != nil {
			t.Fatal(err)
		}
	}
}
