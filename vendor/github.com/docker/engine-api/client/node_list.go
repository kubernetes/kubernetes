package client

import (
	"encoding/json"
	"net/url"

	"github.com/docker/engine-api/types"
	"github.com/docker/engine-api/types/filters"
	"github.com/docker/engine-api/types/swarm"
	"golang.org/x/net/context"
)

// NodeList returns the list of nodes.
func (cli *Client) NodeList(ctx context.Context, options types.NodeListOptions) ([]swarm.Node, error) {
	query := url.Values{}

	if options.Filter.Len() > 0 {
		filterJSON, err := filters.ToParam(options.Filter)

		if err != nil {
			return nil, err
		}

		query.Set("filters", filterJSON)
	}

	resp, err := cli.get(ctx, "/nodes", query, nil)
	if err != nil {
		return nil, err
	}

	var nodes []swarm.Node
	err = json.NewDecoder(resp.body).Decode(&nodes)
	ensureReaderClosed(resp)
	return nodes, err
}
