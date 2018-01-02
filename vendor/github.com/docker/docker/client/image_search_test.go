package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"encoding/json"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/registry"
)

func TestImageSearchAnyError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ImageSearch(context.Background(), "some-image", types.ImageSearchOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImageSearchStatusUnauthorizedError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	_, err := client.ImageSearch(context.Background(), "some-image", types.ImageSearchOptions{})
	if err == nil || err.Error() != "Error response from daemon: Unauthorized error" {
		t.Fatalf("expected an Unauthorized Error, got %v", err)
	}
}

func TestImageSearchWithUnauthorizedErrorAndPrivilegeFuncError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	privilegeFunc := func() (string, error) {
		return "", fmt.Errorf("Error requesting privilege")
	}
	_, err := client.ImageSearch(context.Background(), "some-image", types.ImageSearchOptions{
		PrivilegeFunc: privilegeFunc,
	})
	if err == nil || err.Error() != "Error requesting privilege" {
		t.Fatalf("expected an error requesting privilege, got %v", err)
	}
}

func TestImageSearchWithUnauthorizedErrorAndAnotherUnauthorizedError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	privilegeFunc := func() (string, error) {
		return "a-auth-header", nil
	}
	_, err := client.ImageSearch(context.Background(), "some-image", types.ImageSearchOptions{
		PrivilegeFunc: privilegeFunc,
	})
	if err == nil || err.Error() != "Error response from daemon: Unauthorized error" {
		t.Fatalf("expected an Unauthorized Error, got %v", err)
	}
}

func TestImageSearchWithPrivilegedFuncNoError(t *testing.T) {
	expectedURL := "/images/search"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			auth := req.Header.Get("X-Registry-Auth")
			if auth == "NotValid" {
				return &http.Response{
					StatusCode: http.StatusUnauthorized,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte("Invalid credentials"))),
				}, nil
			}
			if auth != "IAmValid" {
				return nil, fmt.Errorf("Invalid auth header : expected 'IAmValid', got %s", auth)
			}
			query := req.URL.Query()
			term := query.Get("term")
			if term != "some-image" {
				return nil, fmt.Errorf("term not set in URL query properly. Expected 'some-image', got %s", term)
			}
			content, err := json.Marshal([]registry.SearchResult{
				{
					Name: "anything",
				},
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(content)),
			}, nil
		}),
	}
	privilegeFunc := func() (string, error) {
		return "IAmValid", nil
	}
	results, err := client.ImageSearch(context.Background(), "some-image", types.ImageSearchOptions{
		RegistryAuth:  "NotValid",
		PrivilegeFunc: privilegeFunc,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %v", results)
	}
}

func TestImageSearchWithoutErrors(t *testing.T) {
	expectedURL := "/images/search"
	filterArgs := filters.NewArgs()
	filterArgs.Add("is-automated", "true")
	filterArgs.Add("stars", "3")

	expectedFilters := `{"is-automated":{"true":true},"stars":{"3":true}}`

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			query := req.URL.Query()
			term := query.Get("term")
			if term != "some-image" {
				return nil, fmt.Errorf("term not set in URL query properly. Expected 'some-image', got %s", term)
			}
			filters := query.Get("filters")
			if filters != expectedFilters {
				return nil, fmt.Errorf("filters not set in URL query properly. Expected '%s', got %s", expectedFilters, filters)
			}
			content, err := json.Marshal([]registry.SearchResult{
				{
					Name: "anything",
				},
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(content)),
			}, nil
		}),
	}
	results, err := client.ImageSearch(context.Background(), "some-image", types.ImageSearchOptions{
		Filters: filterArgs,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("expected a result, got %v", results)
	}
}
