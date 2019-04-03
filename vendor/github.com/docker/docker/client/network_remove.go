package client // import "github.com/docker/docker/client"

import "context"

// NetworkRemove removes an existent network from the docker host.
func (cli *Client) NetworkRemove(ctx context.Context, networkID string) error {
	resp, err := cli.delete(ctx, "/networks/"+networkID, nil, nil)
	ensureReaderClosed(resp)
	return wrapResponseError(err, resp, "network", networkID)
}
