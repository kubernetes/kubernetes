package client

import (
	"golang.org/x/net/context"
)

// CheckpointDelete deletes the checkpoint with the given name from the given container
func (cli *Client) CheckpointDelete(ctx context.Context, containerID string, checkpointID string) error {
	resp, err := cli.delete(ctx, "/containers/"+containerID+"/checkpoints/"+checkpointID, nil, nil)
	ensureReaderClosed(resp)
	return err
}
