package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io"

	"github.com/docker/docker/api/types/swarm"
)

// NodeInspectWithRaw returns the node information.
func (cli *Client) NodeInspectWithRaw(ctx context.Context, nodeID string) (swarm.Node, []byte, error) {
	if nodeID == "" {
		return swarm.Node{}, nil, objectNotFoundError{object: "node", id: nodeID}
	}
	serverResp, err := cli.get(ctx, "/nodes/"+nodeID, nil, nil)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return swarm.Node{}, nil, wrapResponseError(err, serverResp, "node", nodeID)
	}

	body, err := io.ReadAll(serverResp.body)
	if err != nil {
		return swarm.Node{}, nil, err
	}

	var response swarm.Node
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&response)
	return response, body, err
}
