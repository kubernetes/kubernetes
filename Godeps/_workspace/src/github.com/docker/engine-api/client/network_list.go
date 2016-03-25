package client

import (
	"encoding/json"
	"net/url"

	"github.com/docker/engine-api/types"
	"github.com/docker/engine-api/types/filters"
	"golang.org/x/net/context"
)

// NetworkList returns the list of networks configured in the docker host.
func (cli *Client) NetworkList(ctx context.Context, options types.NetworkListOptions) ([]types.NetworkResource, error) {
	query := url.Values{}
	if options.Filters.Len() > 0 {
		filterJSON, err := filters.ToParam(options.Filters)
		if err != nil {
			return nil, err
		}

		query.Set("filters", filterJSON)
	}
	var networkResources []types.NetworkResource
	resp, err := cli.get(ctx, "/networks", query, nil)
	if err != nil {
		return networkResources, err
	}
	err = json.NewDecoder(resp.body).Decode(&networkResources)
	ensureReaderClosed(resp)
	return networkResources, err
}
