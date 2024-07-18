package client // import "github.com/docker/docker/client"

import (
	"context"
	"net/url"

	"github.com/docker/docker/api/types"
)

// ContainerStats returns near realtime stats for a given container.
// It's up to the caller to close the io.ReadCloser returned.
func (cli *Client) ContainerStats(ctx context.Context, containerID string, stream bool) (types.ContainerStats, error) {
	query := url.Values{}
	query.Set("stream", "0")
	if stream {
		query.Set("stream", "1")
	}

	resp, err := cli.get(ctx, "/containers/"+containerID+"/stats", query, nil)
	if err != nil {
		return types.ContainerStats{}, err
	}

	osType := getDockerOS(resp.header.Get("Server"))
	return types.ContainerStats{Body: resp.body, OSType: osType}, err
}

// ContainerStatsOneShot gets a single stat entry from a container.
// It differs from `ContainerStats` in that the API should not wait to prime the stats
func (cli *Client) ContainerStatsOneShot(ctx context.Context, containerID string) (types.ContainerStats, error) {
	query := url.Values{}
	query.Set("stream", "0")
	query.Set("one-shot", "1")

	resp, err := cli.get(ctx, "/containers/"+containerID+"/stats", query, nil)
	if err != nil {
		return types.ContainerStats{}, err
	}

	osType := getDockerOS(resp.header.Get("Server"))
	return types.ContainerStats{Body: resp.body, OSType: osType}, err
}
