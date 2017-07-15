package client

import (
	"net/url"

	"golang.org/x/net/context"
)

// SwarmLeave leaves the Swarm.
func (cli *Client) SwarmLeave(ctx context.Context, force bool) error {
	query := url.Values{}
	if force {
		query.Set("force", "1")
	}
	resp, err := cli.post(ctx, "/swarm/leave", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
