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

func TestContainerStatsError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerStats(context.Background(), "nothing", false)
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerStats(t *testing.T) {
	expectedURL := "/containers/container_id/stats"
	cases := []struct {
		stream         bool
		expectedStream string
	}{
		{
			expectedStream: "0",
		},
		{
			stream:         true,
			expectedStream: "1",
		},
	}
	for _, c := range cases {
		client := &Client{
			client: newMockClient(func(r *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(r.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, r.URL)
				}

				query := r.URL.Query()
				stream := query.Get("stream")
				if stream != c.expectedStream {
					return nil, fmt.Errorf("stream not set in URL query properly. Expected '%s', got %s", c.expectedStream, stream)
				}

				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte("response"))),
				}, nil
			}),
		}
		resp, err := client.ContainerStats(context.Background(), "container_id", c.stream)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		content, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(content) != "response" {
			t.Fatalf("expected response to contain 'response', got %s", string(content))
		}
	}
}
