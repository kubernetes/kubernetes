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
	"github.com/docker/docker/pkg/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/context"
)

func TestSwarmGetUnlockKeyError(t *testing.T) {
	client := &Client{
		client: newMockClient(errorMock(http.StatusInternalServerError, "Server error")),
	}

	_, err := client.SwarmGetUnlockKey(context.Background())
	testutil.ErrorContains(t, err, "Error response from daemon: Server error")
}

func TestSwarmGetUnlockKey(t *testing.T) {
	expectedURL := "/swarm/unlockkey"
	unlockKey := "SWMKEY-1-y6guTZNTwpQeTL5RhUfOsdBdXoQjiB2GADHSRJvbXeE"

	client := &Client{
		client: newMockClient(func(req *http.Request) (*http.Response, error) {
			if !strings.HasPrefix(req.URL.Path, expectedURL) {
				return nil, fmt.Errorf("Expected URL '%s', got '%s'", expectedURL, req.URL)
			}
			if req.Method != "GET" {
				return nil, fmt.Errorf("expected GET method, got %s", req.Method)
			}

			key := types.SwarmUnlockKeyResponse{
				UnlockKey: unlockKey,
			}

			b, err := json.Marshal(key)
			if err != nil {
				return nil, err
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader(b)),
			}, nil
		}),
	}

	resp, err := client.SwarmGetUnlockKey(context.Background())
	require.NoError(t, err)
	assert.Equal(t, unlockKey, resp.UnlockKey)
}
