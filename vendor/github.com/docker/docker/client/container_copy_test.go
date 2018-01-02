package client

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
)

func TestContainerStatPathError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerStatPath(context.Background(), "container_id", "path")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server error, got %v", err)
	}
}

func TestContainerStatPathNoHeaderError(t *testing.T) {
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}
	_, err := client.ContainerStatPath(context.Background(), "container_id", "path/to/file")
	if err == nil {
		t.Fatalf("expected an error, got nothing")
	}
}

func TestContainerStatPath(t *testing.T) {
	expectedURL := "/containers/container_id/archive"
	expectedPath := "path/to/file"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "HEAD" {
				return nil, fmt.Errorf("expected HEAD method, got %s", req.Method)
			}
			query := req.URL.Query()
			path := query.Get("path")
			if path != expectedPath {
				return nil, fmt.Errorf("path not set in URL query properly")
			}
			content, err := json.Marshal(types.ContainerPathStat{
				Name: "name",
				Mode: 0700,
			})
			if err != nil {
				return nil, err
			}
			base64PathStat := base64.StdEncoding.EncodeToString(content)
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
				Header: http.Header{
					"X-Docker-Container-Path-Stat": []string{base64PathStat},
				},
			}, nil
		}),
	}
	stat, err := client.ContainerStatPath(context.Background(), "container_id", expectedPath)
	if err != nil {
		t.Fatal(err)
	}
	if stat.Name != "name" {
		t.Fatalf("expected container path stat name to be 'name', got '%s'", stat.Name)
	}
	if stat.Mode != 0700 {
		t.Fatalf("expected container path stat mode to be 0700, got '%v'", stat.Mode)
	}
}

func TestCopyToContainerError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	err := client.CopyToContainer(context.Background(), "container_id", "path/to/file", bytes.NewReader([]byte("")), types.CopyToContainerOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server error, got %v", err)
	}
}

func TestCopyToContainerNotStatusOKError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNoContent, "No content")),
	}
	err := client.CopyToContainer(context.Background(), "container_id", "path/to/file", bytes.NewReader([]byte("")), types.CopyToContainerOptions{})
	if err == nil || err.Error() != "unexpected status code from daemon: 204" {
		t.Fatalf("expected an unexpected status code error, got %v", err)
	}
}

func TestCopyToContainer(t *testing.T) {
	expectedURL := "/containers/container_id/archive"
	expectedPath := "path/to/file"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "PUT" {
				return nil, fmt.Errorf("expected PUT method, got %s", req.Method)
			}
			query := req.URL.Query()
			path := query.Get("path")
			if path != expectedPath {
				return nil, fmt.Errorf("path not set in URL query properly, expected '%s', got %s", expectedPath, path)
			}
			noOverwriteDirNonDir := query.Get("noOverwriteDirNonDir")
			if noOverwriteDirNonDir != "true" {
				return nil, fmt.Errorf("noOverwriteDirNonDir not set in URL query properly, expected true, got %s", noOverwriteDirNonDir)
			}

			content, err := ioutil.ReadAll(req.Body)
			if err != nil {
				return nil, err
			}
			if err := req.Body.Close(); err != nil {
				return nil, err
			}
			if string(content) != "content" {
				return nil, fmt.Errorf("expected content to be 'content', got %s", string(content))
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}
	err := client.CopyToContainer(context.Background(), "container_id", expectedPath, bytes.NewReader([]byte("content")), types.CopyToContainerOptions{
		AllowOverwriteDirWithFile: false,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCopyFromContainerError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, _, err := client.CopyFromContainer(context.Background(), "container_id", "path/to/file")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server error, got %v", err)
	}
}

func TestCopyFromContainerNotStatusOKError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNoContent, "No content")),
	}
	_, _, err := client.CopyFromContainer(context.Background(), "container_id", "path/to/file")
	if err == nil || err.Error() != "unexpected status code from daemon: 204" {
		t.Fatalf("expected an unexpected status code error, got %v", err)
	}
}

func TestCopyFromContainerNoHeaderError(t *testing.T) {
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
			}, nil
		}),
	}
	_, _, err := client.CopyFromContainer(context.Background(), "container_id", "path/to/file")
	if err == nil {
		t.Fatalf("expected an error, got nothing")
	}
}

func TestCopyFromContainer(t *testing.T) {
	expectedURL := "/containers/container_id/archive"
	expectedPath := "path/to/file"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "GET" {
				return nil, fmt.Errorf("expected GET method, got %s", req.Method)
			}
			query := req.URL.Query()
			path := query.Get("path")
			if path != expectedPath {
				return nil, fmt.Errorf("path not set in URL query properly, expected '%s', got %s", expectedPath, path)
			}

			headercontent, err := json.Marshal(types.ContainerPathStat{
				Name: "name",
				Mode: 0700,
			})
			if err != nil {
				return nil, err
			}
			base64PathStat := base64.StdEncoding.EncodeToString(headercontent)

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte("content"))),
				Header: http.Header{
					"X-Docker-Container-Path-Stat": []string{base64PathStat},
				},
			}, nil
		}),
	}
	r, stat, err := client.CopyFromContainer(context.Background(), "container_id", expectedPath)
	if err != nil {
		t.Fatal(err)
	}
	if stat.Name != "name" {
		t.Fatalf("expected container path stat name to be 'name', got '%s'", stat.Name)
	}
	if stat.Mode != 0700 {
		t.Fatalf("expected container path stat mode to be 0700, got '%v'", stat.Mode)
	}
	content, err := ioutil.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if string(content) != "content" {
		t.Fatalf("expected content to be 'content', got %s", string(content))
	}
}
