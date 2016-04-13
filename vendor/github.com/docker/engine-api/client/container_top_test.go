package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func TestContainerTopError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ContainerTop(context.Background(), "nothing", []string{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestContainerTop(t *testing.T) {
	expectedProcesses := [][]string{
		{"p1", "p2"},
		{"p3"},
	}
	expectedTitles := []string{"title1", "title2"}

	client := &Client{
		transport: newMockClient(nil, func(req *http.Request) (*http.Response, error) {
			query := req.URL.Query()
			args := query.Get("ps_args")
			if args != "arg1 arg2" {
				return nil, fmt.Errorf("args not set in URL query properly. Expected 'arg1 arg2', got %v", args)
			}

			b, err := json.Marshal(types.ContainerProcessList{
				Processes: [][]string{
					{"p1", "p2"},
					{"p3"},
				},
				Titles: []string{"title1", "title2"},
			})
			if err != nil {
				return nil, err
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}),
	}

	processList, err := client.ContainerTop(context.Background(), "container_id", []string{"arg1", "arg2"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expectedProcesses, processList.Processes) {
		t.Fatalf("Processes: expected %v, got %v", expectedProcesses, processList.Processes)
	}
	if !reflect.DeepEqual(expectedTitles, processList.Titles) {
		t.Fatalf("Titles: expected %v, got %v", expectedTitles, processList.Titles)
	}
}
