package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"golang.org/x/net/context"
)

func TestImageListError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
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
			client: newMockClient(func(req *http.Request) (*http.Response, error) {
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
				content, err := json.Marshal([]types.ImageSummary{
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

func TestImageListApiBefore125(t *testing.T) {
	expectedFilter := "image:tag"
	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			query := req.URL.Query()
			actualFilter := query.Get("filter")
			if actualFilter != expectedFilter {
				return nil, fmt.Errorf("filter not set in URL query properly. Expected '%s', got %s", expectedFilter, actualFilter)
			}
			actualFilters := query.Get("filters")
			if actualFilters != "" {
				return nil, fmt.Errorf("filters should have not been present, were with value: %s", actualFilters)
			}
			content, err := json.Marshal([]types.ImageSummary{
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
		version: "1.24",
	}

	filters := filters.NewArgs()
	filters.Add("reference", "image:tag")

	options := types.ImageListOptions{
		Filters: filters,
	}

	images, err := client.ImageList(context.Background(), options)
	if err != nil {
		t.Fatal(err)
	}
	if len(images) != 2 {
		t.Fatalf("expected 2 images, got %v", images)
	}
}
