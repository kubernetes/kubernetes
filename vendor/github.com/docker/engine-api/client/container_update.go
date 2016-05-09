package client

import (
	"github.com/docker/engine-api/types/container"
	"golang.org/x/net/context"
)

// ContainerUpdate updates resources of a container
func (cli *Client) ContainerUpdate(ctx context.Context, containerID string, updateConfig container.UpdateConfig) error {
	resp, err := cli.post(ctx, "/containers/"+containerID+"/update", nil, updateConfig, nil)
	ensureReaderClosed(resp)
	return err
}
