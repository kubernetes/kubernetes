package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/docker/docker/api/types/container"

	"golang.org/x/net/context"
)

func TestContainerWaitError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	resultC, errC := client.ContainerWait(context.Background(), "nothing", "")
	select {
	case result := <-resultC:
		t.Fatalf("expected to not get a wait result, got %d", result.StatusCode)
	case err := <-errC:
		if err.Error() != "Error response from daemon: Server error" {
			t.Fatalf("expected a Server Error, got %v", err)
		}
	}
}

func TestContainerWait(t *testing.T) {
	expectedURL := "/containers/container_id/wait"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			b, err := json.Marshal(container.ContainerWaitOKBody{
				StatusCode: 15,
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

	resultC, errC := client.ContainerWait(context.Background(), "container_id", "")
	select {
	case err := <-errC:
		t.Fatal(err)
	case result := <-resultC:
		if result.StatusCode != 15 {
			t.Fatalf("expected a status code equal to '15', got %d", result.StatusCode)
		}
	}
}

func ExampleClient_ContainerWait_withTimeout() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	client, _ := NewEnvClient()
	_, errC := client.ContainerWait(ctx, "container_id", "")
	if err := <-errC; err != nil {
		log.Fatal(err)
	}
}
