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
)

func TestContainerExecCreateError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerExecCreate(context.Background(), "container_id", types.ExecConfig{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExecCreate(t *testing.T) {
	expectedURL := "/containers/container_id/exec"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}
			// FIXME validate the content is the given ExecConfig ?
			if err := req.ParseForm(); err != nil {
				return nil, err
			}
			execConfig := &types.ExecConfig{}
			if err := json.NewDecoder(req.Body).Decode(execConfig); err != nil {
				return nil, err
			}
			if execConfig.User != "user" {
				return nil, fmt.Errorf("expected an execConfig with User == 'user', got %v", execConfig)
			}
			b, err := json.Marshal(types.IDResponse{
				ID: "exec_id",
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

	r, err := client.ContainerExecCreate(context.Background(), "container_id", types.ExecConfig{
		User: "user",
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.ID != "exec_id" {
		t.Fatalf("expected `exec_id`, got %s", r.ID)
	}
}

func TestContainerExecStartError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerExecStart(context.Background(), "nothing", types.ExecStartCheck{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExecStart(t *testing.T) {
	expectedURL := "/exec/exec_id/start"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if err := req.ParseForm(); err != nil {
				return nil, err
			}
			execStartCheck := &types.ExecStartCheck{}
			if err := json.NewDecoder(req.Body).Decode(execStartCheck); err != nil {
				return nil, err
			}
			if execStartCheck.Tty || !execStartCheck.Detach {
				return nil, fmt.Errorf("expected execStartCheck{Detach:true,Tty:false}, got %v", execStartCheck)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}

	err := client.ContainerExecStart(context.Background(), "exec_id", types.ExecStartCheck{
		Detach: true,
		Tty:    false,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestContainerExecInspectError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerExecInspect(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExecInspect(t *testing.T) {
	expectedURL := "/exec/exec_id/json"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			b, err := json.Marshal(types.ContainerExecInspect{
				ExecID:      "exec_id",
				ContainerID: "container_id",
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

	inspect, err := client.ContainerExecInspect(context.Background(), "exec_id")
	if err != nil {
		t.Fatal(err)
	}
	if inspect.ExecID != "exec_id" {
		t.Fatalf("expected ExecID to be `exec_id`, got %s", inspect.ExecID)
	}
	if inspect.ContainerID != "container_id" {
		t.Fatalf("expected ContainerID `container_id`, got %s", inspect.ContainerID)
	}
}
