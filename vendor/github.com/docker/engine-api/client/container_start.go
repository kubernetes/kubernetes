package client

import "golang.org/x/net/context"

// ContainerStart sends a request to the docker daemon to start a container.
func (cli *Client) ContainerStart(ctx context.Context, containerID string) error {
	resp, err := cli.post(ctx, "/containers/"+containerID+"/start", nil, nil, nil)
	ensureReaderClosed(resp)
	return err
}
