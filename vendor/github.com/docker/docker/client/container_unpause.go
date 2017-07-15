package client

import "golang.org/x/net/context"

// ContainerUnpause resumes the process execution within a container
func (cli *Client) ContainerUnpause(ctx context.Context, containerID string) error {
	resp, err := cli.post(ctx, "/containers/"+containerID+"/unpause", nil, nil, nil)
	ensureReaderClosed(resp)
	return err
}
