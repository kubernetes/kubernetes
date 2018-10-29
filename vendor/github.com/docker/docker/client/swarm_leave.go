package client

import (
	"context"
	"net/url"
)

// SwarmLeave leaves the swarm.
func (cli *Client) SwarmLeave(ctx context.Context, force bool) error {
	query := url.Values{}
	if force {
		query.Set("force", "1")
	}
	resp, err := cli.post(ctx, "/swarm/leave", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
