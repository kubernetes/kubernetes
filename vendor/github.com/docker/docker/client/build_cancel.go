package client

import (
	"context"
	"net/url"
)

// BuildCancel requests the daemon to cancel ongoing build request
func (cli *Client) BuildCancel(ctx context.Context, id string) error {
	query := url.Values{}
	query.Set("id", id)

	serverResp, err := cli.post(ctx, "/build/cancel", query, nil, nil)
	if err != nil {
		return err
	}
	defer ensureReaderClosed(serverResp)

	return nil
}
