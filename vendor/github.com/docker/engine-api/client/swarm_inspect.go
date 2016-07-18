package client

import (
	"encoding/json"

	"github.com/docker/engine-api/types/swarm"
	"golang.org/x/net/context"
)

// SwarmInspect inspects the Swarm.
func (cli *Client) SwarmInspect(ctx context.Context) (swarm.Swarm, error) {
	serverResp, err := cli.get(ctx, "/swarm", nil, nil)
	if err != nil {
		return swarm.Swarm{}, err
	}

	var response swarm.Swarm
	err = json.NewDecoder(serverResp.body).Decode(&response)
	ensureReaderClosed(serverResp)
	return response, err
}
