package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/engine-api/types"
	"github.com/docker/engine-api/types/filters"
	"golang.org/x/net/context"
)

func TestImageListError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.ImageList(context.Background(), types.ImageListOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestImageList(t *testing.T) {
	expectedURL := "/images/json"

	noDanglingfilters := filters.NewArgs()
	noDanglingfilters.Add("dangling", "false")

	filters := filters.NewArgs()
	filters.Add("label", "label1")
	filters.Add("label", "label2")
	filters.Add("dangling", "true")

	listCases := []struct {
		options             types.ImageListOptions
		expectedQueryParams map[string]string
	}{
		{
			options: types.ImageListOptions{},
			expectedQueryParams: map[string]string{
				"all":     "",
				"filter":  "",
				"filters": "",
			},
		},
		{
			options: types.ImageListOptions{
				All:       true,
				MatchName: "image_name",
			},
			expectedQueryParams: map[string]string{
				"all":     "1",
				"filter":  "image_name",
				"filters": "",
			},
		},
		{
			options: types.ImageListOptions{
				Filters: filters,
			},
			expectedQueryParams: map[string]string{
				"all":     "",
				"filter":  "",
				"filters": `{"dangling":{"true":true},"label":{"label1":true,"label2":true}}`,
			},
		},
		{
			options: types.ImageListOptions{
				Filters: noDanglingfilters,
			},
			expectedQueryParams: map[string]string{
				"all":     "",
				"filter":  "",
				"filters": `{"dangling":{"false":true}}`,
			},
		},
	}
	for _, listCase := range listCases {
		client := &Client{
			transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				query := req.URL.Query()
				for key, expected := range listCase.expectedQueryParams {
					actual := query.Get(key)
					if actual != expected {
						return nil, fmt.Errorf("%s not set in URL query properly. Expected '%s', got %s", key, expected, actual)
					}
				}
				content, err := json.Marshal([]types.Image{
					{
						ID: "image_id2",
					},
					{
						ID: "image_id2",
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

		images, err := client.ImageList(context.Background(), listCase.options)
		if err != nil {
			t.Fatal(err)
		}
		if len(images) != 2 {
			t.Fatalf("expected 2 images, got %v", images)
		}
	}
}
