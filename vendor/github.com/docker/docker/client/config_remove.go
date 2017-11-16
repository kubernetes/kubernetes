package client

import "golang.org/x/net/context"

// ConfigRemove removes a Config.
func (cli *Client) ConfigRemove(ctx context.Context, id string) error {
	if err := cli.NewVersionError("1.30", "config remove"); err != nil {
		return err
	}
	resp, err := cli.delete(ctx, "/configs/"+id, nil, nil)
	ensureReaderClosed(resp)
	return err
}
