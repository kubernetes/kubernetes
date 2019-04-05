package client // import "github.com/docker/docker/client"

import (
	"context"

	"github.com/docker/docker/api/types"
)

// NetworkDisconnect disconnects a container from an existent network in the docker host.
func (cli *Client) NetworkDisconnect(ctx context.Context, networkID, containerID string, force bool) error {
	nd := types.NetworkDisconnect{Container: containerID, Force: force}
	resp, err := cli.post(ctx, "/networks/"+networkID+"/disconnect", nil, nd, nil)
	ensureReaderClosed(resp)
	return err
}
