package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"net/url"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/swarm"
)

// ServiceList returns the list of services.
func (cli *Client) ServiceList(ctx context.Context, options types.ServiceListOptions) ([]swarm.Service, error) {
	query := url.Values{}

	if options.Filters.Len() > 0 {
		filterJSON, err := filters.ToJSON(options.Filters)
		if err != nil {
			return nil, err
		}

		query.Set("filters", filterJSON)
	}

	resp, err := cli.get(ctx, "/services", query, nil)
	if err != nil {
		return nil, err
	}

	var services []swarm.Service
	err = json.NewDecoder(resp.body).Decode(&services)
	ensureReaderClosed(resp)
	return services, err
}
