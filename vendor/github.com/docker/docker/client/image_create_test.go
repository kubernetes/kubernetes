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

func TestImageCreateError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ImageCreate(context.Background(), "reference", types.ImageCreateOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server error, got %v", err)
	}
}

func TestImageCreate(t *testing.T) {
	expectedURL := "/images/create"
	expectedImage := "test:5000/my_image"
	expectedTag := "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
	expectedReference := fmt.Sprintf("%s@%s", expectedImage, expectedTag)
	expectedRegistryAuth := "eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsiYXV0aCI6ImRHOTBid289IiwiZW1haWwiOiJqb2huQGRvZS5jb20ifX0="
	client := &Client{
		client: newMockClient(func(r *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(r.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, r.URL)
			}
			registryAuth := r.Header.Get("X-Registry-Auth")
			if registryAuth != expectedRegistryAuth {
				return nil, fmt.Errorf("X-Registry-Auth header not properly set in the request. Expected '%s', got %s", expectedRegistryAuth, registryAuth)
			}

			query := r.URL.Query()
			fromImage := query.Get("fromImage")
			if fromImage != expectedImage {
				return nil, fmt.Errorf("fromImage not set in URL query properly. Expected '%s', got %s", expectedImage, fromImage)
			}

			tag := query.Get("tag")
			if tag != expectedTag {
				return nil, fmt.Errorf("tag not set in URL query properly. Expected '%s', got %s", expectedTag, tag)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte("body"))),
			}, nil
		}),
	}

	createResponse, err := client.ImageCreate(context.Background(), expectedReference, types.ImageCreateOptions{
		RegistryAuth: expectedRegistryAuth,
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := ioutil.ReadAll(createResponse)
	if err != nil {
		t.Fatal(err)
	}
	if err = createResponse.Close(); err != nil {
		t.Fatal(err)
	}
	if string(response) != "body" {
		t.Fatalf("expected Body to contain 'body' string, got %s", response)
	}
}
