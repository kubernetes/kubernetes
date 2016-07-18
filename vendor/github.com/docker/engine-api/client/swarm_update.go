package client

import (
	"net/url"
	"strconv"

	"github.com/docker/engine-api/types/swarm"
	"golang.org/x/net/context"
)

// SwarmUpdate updates the Swarm.
func (cli *Client) SwarmUpdate(ctx context.Context, version swarm.Version, swarm swarm.Spec) error {
	query := url.Values{}
	query.Set("version", strconv.FormatUint(version.Index, 10))
	resp, err := cli.post(ctx, "/swarm/update", query, swarm, nil)
	ensureReaderClosed(resp)
	return err
}
