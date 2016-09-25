package client

import (
	"encoding/json"
	"net/http"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// NetworkInspect returns the information for a specific network configured in the docker host.
func (cli *Client) NetworkInspect(ctx context.Context, networkID string) (types.NetworkResource, error) {
	var networkResource types.NetworkResource
	resp, err := cli.get(ctx, "/networks/"+networkID, nil, nil)
	if err != nil {
		if resp.statusCode == http.StatusNotFound {
			return networkResource, networkNotFoundError{networkID}
		}
		return networkResource, err
	}
	err = json.NewDecoder(resp.body).Decode(&networkResource)
	ensureReaderClosed(resp)
	return networkResource, err
}
