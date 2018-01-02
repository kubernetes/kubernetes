package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/events"
	"github.com/docker/docker/api/types/filters"
)

func TestEventsErrorInOptions(t *testing.T) {
	errorCases := []struct {
		options       types.EventsOptions
		expectedError string
	}{
		{
			options: types.EventsOptions{
				Since: "2006-01-02TZ",
			},
			expectedError: `parsing time "2006-01-02TZ"`,
		},
		{
			options: types.EventsOptions{
				Until: "2006-01-02TZ",
			},
			expectedError: `parsing time "2006-01-02TZ"`,
		},
	}
	for _, e := range errorCases {
		client := &Client{
			client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
		}
		_, errs := client.Events(context.Background(), e.options)
		err := <-errs
		if err == nil || !strings.Contains(err.Error(), e.expectedError) {
			t.Fatalf("expected an error %q, got %v", e.expectedError, err)
		}
	}
}

func TestEventsErrorFromServer(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, errs := client.Events(context.Background(), types.EventsOptions{})
	err := <-errs
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestEvents(t *testing.T) {

	expectedURL := "/events"

	filters := filters.NewArgs()
	filters.Add("type", events.ContainerEventType)
	expectedFiltersJSON := fmt.Sprintf(`{"type":{"%s":true}}`, events.ContainerEventType)

	eventsCases := []struct {
		options             types.EventsOptions
		events              []events.Message
		expectedEvents      map[string]bool
		expectedQueryParams map[string]string
	}{
		{
			options: types.EventsOptions{
				Filters: filters,
			},
			expectedQueryParams: map[string]string{
				"filters": expectedFiltersJSON,
			},
			events:         []events.Message{},
			expectedEvents: make(map[string]bool),
		},
		{
			options: types.EventsOptions{
				Filters: filters,
			},
			expectedQueryParams: map[string]string{
				"filters": expectedFiltersJSON,
			},
			events: []events.Message{
				{
					Type:   "container",
					ID:     "1",
					Action: "create",
				},
				{
					Type:   "container",
					ID:     "2",
					Action: "die",
				},
				{
					Type:   "container",
					ID:     "3",
					Action: "create",
				},
			},
			expectedEvents: map[string]bool{
				"1": true,
				"2": true,
				"3": true,
			},
		},
	}

	for _, eventsCase := range eventsCases {
		client := &Client{
			client: newMockClient(func(req *http.Request) (*http.Response, error) {
				if !strings.HasPrefix(req.URL.Path, expectedURL) {
					return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
				}
				query := req.URL.Query()

				for key, expected := range eventsCase.expectedQueryParams {
					actual := query.Get(key)
					if actual != expected {
						return nil, fmt.Errorf("%s not set in URL query properly. Expected '%s', got %s", key, expected, actual)
					}
				}

				buffer := new(bytes.Buffer)

				for _, e := range eventsCase.events {
					b, _ := json.Marshal(e)
					buffer.Write(b)
				}

				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       ioutil.NopCloser(buffer),
				}, nil
			}),
		}

		messages, errs := client.Events(context.Background(), eventsCase.options)

	loop:
		for {
			select {
			case err := <-errs:
				if err != nil && err != io.EOF {
					t.Fatal(err)
				}

				break loop
			case e := <-messages:
				_, ok := eventsCase.expectedEvents[e.ID]
				if !ok {
					t.Fatalf("event received not expected with action %s & id %s", e.Action, e.ID)
				}
			}
		}
	}
}
