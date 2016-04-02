package client

import (
	"io"
	"net/url"

	"golang.org/x/net/context"

	"github.com/docker/engine-api/types"
)

// ImageCreate creates a new image based in the parent options.
// It returns the JSON content in the response body.
func (cli *Client) ImageCreate(ctx context.Context, options types.ImageCreateOptions) (io.ReadCloser, error) {
	query := url.Values{}
	query.Set("fromImage", options.Parent)
	query.Set("tag", options.Tag)
	resp, err := cli.tryImageCreate(ctx, query, options.RegistryAuth)
	if err != nil {
		return nil, err
	}
	return resp.body, nil
}

func (cli *Client) tryImageCreate(ctx context.Context, query url.Values, registryAuth string) (*serverResponse, error) {
	headers := map[string][]string{"X-Registry-Auth": {registryAuth}}
	return cli.post(ctx, "/images/create", query, nil, headers)
}
