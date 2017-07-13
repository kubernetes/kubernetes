package client

import (
	"github.com/docker/docker/api/types/swarm"
	"golang.org/x/net/context"
)

// SwarmUnlock unlockes locked swarm.
func (cli *Client) SwarmUnlock(ctx context.Context, req swarm.UnlockRequest) error {
	serverResp, err := cli.post(ctx, "/swarm/unlock", nil, req, nil)
	if err != nil {
		return err
	}

	ensureReaderClosed(serverResp)
	return err
}
