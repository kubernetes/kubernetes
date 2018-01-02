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

func TestImagePushReferenceError(t *testing.T) {
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			return nil, nil
		}),
	}
	// An empty reference is an invalid reference
	_, err := client.ImagePush(context.Background(), "", types.ImagePushOptions{})
	if err == nil || !strings.Contains(err.Error(), "invalid reference format") {
		t.Fatalf("expected an error, got %v", err)
	}
	// An canonical reference cannot be pushed
	_, err = client.ImagePush(context.Background(), "repo@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", types.ImagePushOptions{})
	if err == nil || err.Error() != "cannot push a digest reference" {
		t.Fatalf("expected an error, got %v", err)
	}
}

func TestImagePushAnyError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ImagePush(context.Background(), "myimage", types.ImagePushOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImagePushStatusUnauthorizedError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	_, err := client.ImagePush(context.Background(), "myimage", types.ImagePushOptions{})
	if err == nil || err.Error() != "Error response from daemon: Unauthorized error" {
		t.Fatalf("expected an Unauthorized Error, got %v", err)
	}
}

func TestImagePushWithUnauthorizedErrorAndPrivilegeFuncError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	privilegeFunc := func() (string, error) {
		return "", fmt.Errorf("Error requesting privilege")
	}
	_, err := client.ImagePush(context.Background(), "myimage", types.ImagePushOptions{
		PrivilegeFunc: privilegeFunc,
	})
	if err == nil || err.Error() != "Error requesting privilege" {
		t.Fatalf("expected an error requesting privilege, got %v", err)
	}
}

func TestImagePushWithUnauthorizedErrorAndAnotherUnauthorizedError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusUnauthorized, "Unauthorized error")),
	}
	privilegeFunc := func() (string, error) {
		return "a-auth-header", nil
	}
	_, err := client.ImagePush(context.Background(), "myimage", types.ImagePushOptions{
		PrivilegeFunc: privilegeFunc,
	})
	if err == nil || err.Error() != "Error response from daemon: Unauthorized error" {
		t.Fatalf("expected an Unauthorized Error, got %v", err)
	}
}

func TestImagePushWithPrivilegedFuncNoError(t *testing.T) {
	expectedURL := "/images/myimage/push"
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
			tag := query.Get("tag")
			if tag != "tag" {
				return nil, fmt.Errorf("tag not set in URL query properly. Expected '%s', got %s", "tag", tag)
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
	resp, err := client.ImagePush(context.Background(), "myimage:tag", types.ImagePushOptions{
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

func TestImagePushWithoutErrors(t *testing.T) {
	expectedOutput := "hello world"
	expectedURLFormat := "/images/%s/push"
	pullCases := []struct {
		reference     string
		expectedImage string
		expectedTag   string
	}{
		{
			reference:     "myimage",
			expectedImage: "myimage",
			expectedTag:   "",
		},
		{
			reference:     "myimage:tag",
			expectedImage: "myimage",
			expectedTag:   "tag",
		},
	}
	for _, pullCase := range pullCases {
		client := &Client{
			client: newMockClient(func(req *http.Request) (*http.Response, error) {
				expectedURL := fmt.Sprintf(expectedURLFormat, pullCase.expectedImage)
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				query := req.URL.Query()
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
		resp, err := client.ImagePush(context.Background(), pullCase.reference, types.ImagePushOptions{})
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
