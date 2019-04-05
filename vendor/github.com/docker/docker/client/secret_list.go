package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"net/url"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/swarm"
)

// SecretList returns the list of secrets.
func (cli *Client) SecretList(ctx context.Context, options types.SecretListOptions) ([]swarm.Secret, error) {
	if err := cli.NewVersionError("1.25", "secret list"); err != nil {
		return nil, err
	}
	query := url.Values{}

	if options.Filters.Len() > 0 {
		filterJSON, err := filters.ToJSON(options.Filters)
		if err != nil {
			return nil, err
		}

		query.Set("filters", filterJSON)
	}

	resp, err := cli.get(ctx, "/secrets", query, nil)
	if err != nil {
		return nil, err
	}

	var secrets []swarm.Secret
	err = json.NewDecoder(resp.body).Decode(&secrets)
	ensureReaderClosed(resp)
	return secrets, err
}
