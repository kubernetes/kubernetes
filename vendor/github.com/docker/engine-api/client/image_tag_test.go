package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func TestImageTagError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}

	err := client.ImageTag(context.Background(), types.ImageTagOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImageTag(t *testing.T) {
	expectedURL := "/images/image_id/tag"
	tagCases := []struct {
		force               bool
		repositoryName      string
		tag                 string
		expectedQueryParams map[string]string
	}{
		{
			force:          false,
			repositoryName: "repository",
			tag:            "tag1",
			expectedQueryParams: map[string]string{
				"force": "",
				"repo":  "repository",
				"tag":   "tag1",
			},
		}, {
			force:          true,
			repositoryName: "another_repository",
			tag:            "latest",
			expectedQueryParams: map[string]string{
				"force": "1",
				"repo":  "another_repository",
				"tag":   "latest",
			},
		},
	}
	for _, tagCase := range tagCases {
		client := &Client{
			transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				if req.Method != "POST" {
					return nil, fmt.Errorf("expected POST method, got %s", req.Method)
				}
				query := req.URL.Query()
				for key, expected := range tagCase.expectedQueryParams {
					actual := query.Get(key)
					if actual != expected {
						return nil, fmt.Errorf("%s not set in URL query properly. Expected '%s', got %s", key, expected, actual)
					}
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
				}, nil
			}),
		}
		err := client.ImageTag(context.Background(), types.ImageTagOptions{
			ImageID:        "image_id",
			Force:          tagCase.force,
			RepositoryName: tagCase.repositoryName,
			Tag:            tagCase.tag,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}
