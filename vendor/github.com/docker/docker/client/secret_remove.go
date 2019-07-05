package client // import "github.com/docker/docker/client"

import "context"

// SecretRemove removes a Secret.
func (cli *Client) SecretRemove(ctx context.Context, id string) error {
	if err := cli.NewVersionError("1.25", "secret remove"); err != nil {
		return err
	}
	resp, err := cli.delete(ctx, "/secrets/"+id, nil, nil)
	defer ensureReaderClosed(resp)
	return wrapResponseError(err, resp, "secret", id)
}
