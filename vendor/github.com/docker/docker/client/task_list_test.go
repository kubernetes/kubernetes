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
	"golang.org/x/net/context"
)

func TestTaskListError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.TaskList(context.Background(), types.TaskListOptions{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestTaskList(t *testing.T) {
	expectedURL := "/tasks"

	filters := filters.NewArgs()
	filters.Add("label", "label1")
	filters.Add("label", "label2")

	listCases := []struct {
		options             types.TaskListOptions
		expectedQueryParams map[string]string
	}{
		{
			options: types.TaskListOptions{},
			expectedQueryParams: map[string]string{
				"filters": "",
			},
		},
		{
			options: types.TaskListOptions{
				Filters: filters,
			},
			expectedQueryParams: map[string]string{
				"filters": `{"label":{"label1":true,"label2":true}}`,
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
				content, err := json.Marshal([]swarm.Task{
					{
						ID: "task_id1",
					},
					{
						ID: "task_id2",
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

		tasks, err := client.TaskList(context.Background(), listCase.options)
		if err != nil {
			t.Fatal(err)
		}
		if len(tasks) != 2 {
			t.Fatalf("expected 2 tasks, got %v", tasks)
		}
	}
}
