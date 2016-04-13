package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func TestImageInspectError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, _, err := client.ImageInspectWithRaw(context.Background(), "nothing", true)
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImageInspectImageNotFound(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusNotFound, "Server error")),
	}

	_, _, err := client.ImageInspectWithRaw(context.Background(), "unknown", true)
	if err == nil || !IsErrImageNotFound(err) {
		t.Fatalf("expected a imageNotFound error, got %v", err)
	}
}

func TestImageInspect(t *testing.T) {
	expectedURL := "/images/image_id/json"
	expectedTags := []string{"tag1", "tag2"}
	inspectCases := []struct {
		size                bool
		expectedQueryParams map[string]string
	}{
		{
			size: true,
			expectedQueryParams: map[string]string{
				"size": "1",
			},
		},
		{
			size: false,
			expectedQueryParams: map[string]string{
				"size": "",
			},
		},
	}
	for _, inspectCase := range inspectCases {
		client := &Client{
			transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				query := req.URL.Query()
				for key, expected := range inspectCase.expectedQueryParams {
					actual := query.Get(key)
					if actual != expected {
						return nil, fmt.Errorf("%s not set in URL query properly. Expected '%s', got %s", key, expected, actual)
					}
				}
				content, err := json.Marshal(types.ImageInspect{
					ID:       "image_id",
					RepoTags: expectedTags,
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

		imageInspect, _, err := client.ImageInspectWithRaw(context.Background(), "image_id", inspectCase.size)
		if err != nil {
			t.Fatal(err)
		}
		if imageInspect.ID != "image_id" {
			t.Fatalf("expected `image_id`, got %s", imageInspect.ID)
		}
		if !reflect.DeepEqual(imageInspect.RepoTags, expectedTags) {
			t.Fatalf("expected `%v`, got %v", expectedTags, imageInspect.RepoTags)
		}
	}
}
