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
	"github.com/docker/docker/api/types/swarm"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestSecretListUnsupported(t *testing.T) {
	client := &Client{
		version: "1.24",
		client:  &http.Client{},
	}
	_, err := client.SecretList(context.Background(), types.SecretListOptions{})
	assert.EqualError(t, err, `"secret list" requires API version 1.25, but the Docker daemon API version is 1.24`)
}

func TestSecretListError(t *testing.T) {
	client := &Client{
		version: "1.25",
		client:  newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.SecretList(context.Background(), types.SecretListOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestSecretList(t *testing.T) {
	expectedURL := "/v1.25/secrets"

	filters := filters.NewArgs()
	filters.Add("label", "label1")
	filters.Add("label", "label2")

	listCases := []struct {
		options             types.SecretListOptions
		expectedQueryParams map[string]string
	}{
		{
			options: types.SecretListOptions{},
			expectedQueryParams: map[string]string{
				"filters": "",
			},
		},
		{
			options: types.SecretListOptions{
				Filters: filters,
			},
			expectedQueryParams: map[string]string{
				"filters": `{"label":{"label1":true,"label2":true}}`,
			},
		},
	}
	for _, listCase := range listCases {
		client := &Client{
			version: "1.25",
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
				content, err := json.Marshal([]swarm.Secret{
					{
						ID: "secret_id1",
					},
					{
						ID: "secret_id2",
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

		secrets, err := client.SecretList(context.Background(), listCase.options)
		if err != nil {
			t.Fatal(err)
		}
		if len(secrets) != 2 {
			t.Fatalf("expected 2 secrets, got %v", secrets)
		}
	}
}
