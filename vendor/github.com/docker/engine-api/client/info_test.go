package client

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

func infoMock(req *http.Request) (*http.Response, error) {
	info := &types.Info{
		ID:         "daemonID",
		Containers: 3,
	}
	b, err := json.Marshal(info)
	if err != nil {
		return nil, err
	}

	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader(b)),
	}, nil
}

func TestInfo(t *testing.T) {
	client := &Client{
		transport: newMockClient(nil, infoMock),
	}

	info, err := client.Info(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	if info.ID != "daemonID" {
		t.Fatalf("expected daemonID, got %s", info.ID)
	}

	if info.Containers != 3 {
		t.Fatalf("expected 3 containers, got %d", info.Containers)
	}
}
