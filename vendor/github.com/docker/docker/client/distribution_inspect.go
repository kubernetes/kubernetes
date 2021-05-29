package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"net/url"

	registrytypes "github.com/docker/docker/api/types/registry"
)

// DistributionInspect returns the image digest with full Manifest
func (cli *Client) DistributionInspect(ctx context.Context, image, encodedRegistryAuth string) (registrytypes.DistributionInspect, error) {
	// Contact the registry to retrieve digest and platform information
	var distributionInspect registrytypes.DistributionInspect
	if image == "" {
		return distributionInspect, objectNotFoundError{object: "distribution", id: image}
	}

	if err := cli.NewVersionError("1.30", "distribution inspect"); err != nil {
		return distributionInspect, err
	}
	var headers map[string][]string

	if encodedRegistryAuth != "" {
		headers = map[string][]string{
			"X-Registry-Auth": {encodedRegistryAuth},
		}
	}

	resp, err := cli.get(ctx, "/distribution/"+image+"/json", url.Values{}, headers)
	defer ensureReaderClosed(resp)
	if err != nil {
		return distributionInspect, err
	}

	err = json.NewDecoder(resp.body).Decode(&distributionInspect)
	return distributionInspect, err
}
