package client

import (
	"encoding/json"
	"net/url"

	"github.com/docker/engine-api/types"
	"github.com/docker/engine-api/types/filters"
	"golang.org/x/net/context"
)

// VolumeList returns the volumes configured in the docker host.
func (cli *Client) VolumeList(ctx context.Context, filter filters.Args) (types.VolumesListResponse, error) {
	var volumes types.VolumesListResponse
	query := url.Values{}

	if filter.Len() > 0 {
		filterJSON, err := filters.ToParamWithVersion(cli.version, filter)
		if err != nil {
			return volumes, err
		}
		query.Set("filters", filterJSON)
	}
	resp, err := cli.get(ctx, "/volumes", query, nil)
	if err != nil {
		return volumes, err
	}

	err = json.NewDecoder(resp.body).Decode(&volumes)
	ensureReaderClosed(resp)
	return volumes, err
}
