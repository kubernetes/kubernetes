package client

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func TestContainerDiffError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerDiff(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}

}

func TestContainerDiff(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
			b, err := json.Marshal([]types.ContainerChange{
				{
					Kind: 0,
					Path: "/path/1",
				},
				{
					Kind: 1,
					Path: "/path/2",
				},
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}),
	}

	changes, err := client.ContainerDiff(context.Background(), "container_id")
	if err != nil {
		t.Fatal(err)
	}
	if len(changes) != 2 {
		t.Fatalf("expected an array of 2 changes, got %v", changes)
	}
}
