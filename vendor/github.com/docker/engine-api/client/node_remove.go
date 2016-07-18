package client

import "golang.org/x/net/context"

// NodeRemove removes a Node.
func (cli *Client) NodeRemove(ctx context.Context, nodeID string) error {
	resp, err := cli.delete(ctx, "/nodes/"+nodeID, nil, nil)
	ensureReaderClosed(resp)
	return err
}
