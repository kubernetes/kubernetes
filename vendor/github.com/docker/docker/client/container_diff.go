package client

import (
	"encoding/json"
	"net/url"

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

// ContainerDiff shows differences in a container filesystem since it was started.
func (cli *Client) ContainerDiff(ctx context.Context, containerID string) ([]types.ContainerChange, error) {
	var changes []types.ContainerChange

	serverResp, err := cli.get(ctx, "/containers/"+containerID+"/changes", url.Values{}, nil)
	if err != nil {
		return changes, err
	}

	err = json.NewDecoder(serverResp.body).Decode(&changes)
	ensureReaderClosed(serverResp)
	return changes, err
}
