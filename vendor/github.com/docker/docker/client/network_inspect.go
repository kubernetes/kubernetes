package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io/ioutil"
	"net/url"

	"github.com/docker/docker/api/types"
)

// NetworkInspect returns the information for a specific network configured in the docker host.
func (cli *Client) NetworkInspect(ctx context.Context, networkID string, options types.NetworkInspectOptions) (types.NetworkResource, error) {
	networkResource, _, err := cli.NetworkInspectWithRaw(ctx, networkID, options)
	return networkResource, err
}

// NetworkInspectWithRaw returns the information for a specific network configured in the docker host and its raw representation.
func (cli *Client) NetworkInspectWithRaw(ctx context.Context, networkID string, options types.NetworkInspectOptions) (types.NetworkResource, []byte, error) {
	if networkID == "" {
		return types.NetworkResource{}, nil, objectNotFoundError{object: "network", id: networkID}
	}
	var (
		networkResource types.NetworkResource
		resp            serverResponse
		err             error
	)
	query := url.Values{}
	if options.Verbose {
		query.Set("verbose", "true")
	}
	if options.Scope != "" {
		query.Set("scope", options.Scope)
	}
	resp, err = cli.get(ctx, "/networks/"+networkID, query, nil)
	defer ensureReaderClosed(resp)
	if err != nil {
		return networkResource, nil, wrapResponseError(err, resp, "network", networkID)
	}

	body, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return networkResource, nil, err
	}
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&networkResource)
	return networkResource, body, err
}
