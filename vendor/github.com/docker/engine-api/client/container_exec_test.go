package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func TestContainerExecCreateError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerExecCreate(context.Background(), types.ExecConfig{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExecCreate(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
			// FIXME validate the content is the given ExecConfig ?
			if err := req.ParseForm(); err != nil {
				return nil, err
			}
			execConfig := &types.ExecConfig{}
			if err := json.NewDecoder(req.Body).Decode(execConfig); err != nil {
				return nil, err
			}
			if execConfig.Container != "container_id" {
				return nil, fmt.Errorf("expected an execConfig with Container == 'container_id', got %v", execConfig)
			}
			b, err := json.Marshal(types.ContainerExecCreateResponse{
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

	r, err := client.ContainerExecCreate(context.Background(), types.ExecConfig{
		Container: "container_id",
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
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.ContainerExecStart(context.Background(), "nothing", types.ExecStartCheck{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExecStart(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
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
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerExecInspect(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerExecInspect(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
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
