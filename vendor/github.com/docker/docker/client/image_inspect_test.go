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

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

func TestImageInspectError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, _, err := client.ImageInspectWithRaw(context.Background(), "nothing")
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImageInspectImageNotFound(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusNotFound, "Server error")),
	}

	_, _, err := client.ImageInspectWithRaw(context.Background(), "unknown")
	if err == nil || !IsErrImageNotFound(err) {
		t.Fatalf("expected an imageNotFound error, got %v", err)
	}
}

func TestImageInspect(t *testing.T) {
	expectedURL := "/images/image_id/json"
	expectedTags := []string{"tag1", "tag2"}
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
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

	imageInspect, _, err := client.ImageInspectWithRaw(context.Background(), "image_id")
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
