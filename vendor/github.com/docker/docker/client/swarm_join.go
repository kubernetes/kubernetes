package client

import (
	"github.com/docker/docker/api/types/swarm"
	"golang.org/x/net/context"
)

// SwarmJoin joins the Swarm.
func (cli *Client) SwarmJoin(ctx context.Context, req swarm.JoinRequest) error {
	resp, err := cli.post(ctx, "/swarm/join", nil, req, nil)
	ensureReaderClosed(resp)
	return err
}
