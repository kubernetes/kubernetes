package client // import "github.com/docker/docker/client"

import (
	"context"
	"net/url"
)

// ContainerKill terminates the container process but does not remove the container from the docker host.
func (cli *Client) ContainerKill(ctx context.Context, containerID, signal string) error {
	query := url.Values{}
	query.Set("signal", signal)

	resp, err := cli.post(ctx, "/containers/"+containerID+"/kill", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
