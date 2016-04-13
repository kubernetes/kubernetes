package client

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"

	"golang.org/x/net/context"

	"strings"
)

func TestImageSaveError(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, errorMock(http.StatusInternalServerError, "Server error")),
	}
	_, err := client.ImageSave(context.Background(), []string{"nothing"})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server error, got %v", err)
	}
}

func TestImageSave(t *testing.T) {
	expectedURL := "/images/get"
	client := &Client{
		transport: newMockClient(nil, func(r *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(r.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, r.URL)
			}
			query := r.URL.Query()
			names := query["names"]
			expectedNames := []string{"image_id1", "image_id2"}
			if !reflect.DeepEqual(names, expectedNames) {
				return nil, fmt.Errorf("names not set in URL query properly. Expected %v, got %v", names, expectedNames)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte("response"))),
			}, nil
		}),
	}
	saveResponse, err := client.ImageSave(context.Background(), []string{"image_id1", "image_id2"})
	if err != nil {
		t.Fatal(err)
	}
	response, err := ioutil.ReadAll(saveResponse)
	if err != nil {
		t.Fatal(err)
	}
	saveResponse.Close()
	if string(response) != "response" {
		t.Fatalf("expected response to contain 'response', got %s", string(response))
	}
}
