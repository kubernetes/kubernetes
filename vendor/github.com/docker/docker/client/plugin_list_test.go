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

func TestPluginListError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.PluginList(context.Background(), filters.NewArgs())
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestPluginList(t *testing.T) {
	expectedURL := "/plugins"

	enabledFilters := filters.NewArgs()
	enabledFilters.Add("enabled", "true")

	capabilityFilters := filters.NewArgs()
	capabilityFilters.Add("capability", "volumedriver")
	capabilityFilters.Add("capability", "authz")

	listCases := []struct {
		filters             filters.Args
		expectedQueryParams map[string]string
	}{
		{
			filters: filters.NewArgs(),
			expectedQueryParams: map[string]string{
				"all":     "",
				"filter":  "",
				"filters": "",
			},
		},
		{
			filters: enabledFilters,
			expectedQueryParams: map[string]string{
				"all":     "",
				"filter":  "",
				"filters": `{"enabled":{"true":true}}`,
			},
		},
		{
			filters: capabilityFilters,
			expectedQueryParams: map[string]string{
				"all":     "",
				"filter":  "",
				"filters": `{"capability":{"authz":true,"volumedriver":true}}`,
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
				content, err := json.Marshal([]*types.Plugin{
					{
						ID: "plugin_id1",
					},
					{
						ID: "plugin_id2",
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

		plugins, err := client.PluginList(context.Background(), listCase.filters)
		if err != nil {
			t.Fatal(err)
		}
		if len(plugins) != 2 {
			t.Fatalf("expected 2 plugins, got %v", plugins)
		}
	}
}
