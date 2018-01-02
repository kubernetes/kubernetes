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
	volumetypes "github.com/docker/docker/api/types/volume"
	"golang.org/x/net/context"
)

func TestVolumeCreateError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.VolumeCreate(context.Background(), volumetypes.VolumesCreateBody{})
	if err == nil || err.Error() != "Error response from daemon: Server error" {
		t.Fatalf("expected a Server Error, got %v", err)
	}
}

func TestVolumeCreate(t *testing.T) {
	expectedURL := "/volumes/create"

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}

			if req.Method != "POST" {
				return nil, fmt.Errorf("expected POST method, got %s", req.Method)
			}

			content, err := json.Marshal(types.Volume{
				Name:       "volume",
				Driver:     "local",
				Mountpoint: "mountpoint",
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

	volume, err := client.VolumeCreate(context.Background(), volumetypes.VolumesCreateBody{
		Name:   "myvolume",
		Driver: "mydriver",
		DriverOpts: map[string]string{
			"opt-key": "opt-value",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if volume.Name != "volume" {
		t.Fatalf("expected volume.Name to be 'volume', got %s", volume.Name)
	}
	if volume.Driver != "local" {
		t.Fatalf("expected volume.Driver to be 'local', got %s", volume.Driver)
	}
	if volume.Mountpoint != "mountpoint" {
		t.Fatalf("expected volume.Mountpoint to be 'mountpoint', got %s", volume.Mountpoint)
	}
}
