package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"net/url"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/swarm"
)

// NodeList returns the list of nodes.
func (cli *Client) NodeList(ctx context.Context, options types.NodeListOptions) ([]swarm.Node, error) {
	query := url.Values{}

	if options.Filters.Len() > 0 {
		filterJSON, err := filters.ToJSON(options.Filters)

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
