package client

import "golang.org/x/net/context"

// NetworkRemove removes an existent network from the docker host.
func (cli *Client) NetworkRemove(ctx context.Context, networkID string) error {
	resp, err := cli.delete(ctx, "/networks/"+networkID, nil, nil)
	ensureReaderClosed(resp)
	return err
}
