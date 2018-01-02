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
)

func TestImagePullReferenceParseError(t *testing.T) {
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			return nil, nil
		}),
	}
	// An empty reference is an invalid reference
	_, err := client.ImagePull(context.Background(), "", types.ImagePullOptions{})
	if err == nil || !strings.Contains(err.Error(), "invalid reference format") {
		t.Fatalf("expected an error, got %v", err)
	}
}

func TestImagePullAnyError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ImagePull(context.Background(), "myimage", types.ImagePullOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImagePullStatusUnauthorizedError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	_, err := client.ImagePull(context.Background(), "myimage", types.ImagePullOptions{})
	if err == nil || err.Error() != "Error response from daemon: Unauthorized error" {
		t.Fatalf("expected an Unauthorized Error, got %v", err)
	}
}

func TestImagePullWithUnauthorizedErrorAndPrivilegeFuncError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	privilegeFunc := func() (string, error) {
		return "", fmt.Errorf("Error requesting privilege")
	}
	_, err := client.ImagePull(context.Background(), "myimage", types.ImagePullOptions{
		PrivilegeFunc: privilegeFunc,
	})
	if err == nil || err.Error() != "Error requesting privilege" {
		t.Fatalf("expected an error requesting privilege, got %v", err)
	}
}

func TestImagePullWithUnauthorizedErrorAndAnotherUnauthorizedError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	privilegeFunc := func() (string, error) {
		return "a-auth-header", nil
	}
	_, err := client.ImagePull(context.Background(), "myimage", types.ImagePullOptions{
		PrivilegeFunc: privilegeFunc,
	})
	if err == nil || err.Error() != "Error response from daemon: Unauthorized error" {
		t.Fatalf("expected an Unauthorized Error, got %v", err)
	}
}

func TestImagePullWithPrivilegedFuncNoError(t *testing.T) {
	expectedURL := "/images/create"
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
				return nil, fmt.Errorf("Invalid auth header : expected %s, got %s", "IAmValid", auth)
			}
			query := req.URL.Query()
			fromImage := query.Get("fromImage")
			if fromImage != "myimage" {
				return nil, fmt.Errorf("fromimage not set in URL query properly. Expected '%s', got %s", "myimage", fromImage)
			}
			tag := query.Get("tag")
			if tag != "latest" {
				return nil, fmt.Errorf("tag not set in URL query properly. Expected '%s', got %s", "latest", tag)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte("hello world"))),
			}, nil
		}),
	}
	privilegeFunc := func() (string, error) {
		return "IAmValid", nil
	}
	resp, err := client.ImagePull(context.Background(), "myimage", types.ImagePullOptions{
		RegistryAuth:  "NotValid",
		PrivilegeFunc: privilegeFunc,
	})
	if err != nil {
		t.Fatal(err)
	}
	body, err := ioutil.ReadAll(resp)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "hello world" {
		t.Fatalf("expected 'hello world', got %s", string(body))
	}
}

func TestImagePullWithoutErrors(t *testing.T) {
	expectedURL := "/images/create"
	expectedOutput := "hello world"
	pullCases := []struct {
		all           bool
		reference     string
		expectedImage string
		expectedTag   string
	}{
		{
			all:           false,
			reference:     "myimage",
			expectedImage: "myimage",
			expectedTag:   "latest",
		},
		{
			all:           false,
			reference:     "myimage:tag",
			expectedImage: "myimage",
			expectedTag:   "tag",
		},
		{
			all:           true,
			reference:     "myimage",
			expectedImage: "myimage",
			expectedTag:   "",
		},
		{
			all:           true,
			reference:     "myimage:anything",
			expectedImage: "myimage",
			expectedTag:   "",
		},
	}
	for _, pullCase := range pullCases {
		client := &Client{
			client: newMockClient(func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				query := req.URL.Query()
				fromImage := query.Get("fromImage")
				if fromImage != pullCase.expectedImage {
					return nil, fmt.Errorf("fromimage not set in URL query properly. Expected '%s', got %s", pullCase.expectedImage, fromImage)
				}
				tag := query.Get("tag")
				if tag != pullCase.expectedTag {
					return nil, fmt.Errorf("tag not set in URL query properly. Expected '%s', got %s", pullCase.expectedTag, tag)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(expectedOutput))),
				}, nil
			}),
		}
		resp, err := client.ImagePull(context.Background(), pullCase.reference, types.ImagePullOptions{
			All: pullCase.all,
		})
		if err != nil {
			t.Fatal(err)
		}
		body, err := ioutil.ReadAll(resp)
		if err != nil {
			t.Fatal(err)
		}
		if string(body) != expectedOutput {
			t.Fatalf("expected '%s', got %s", expectedOutput, string(body))
		}
	}
}
