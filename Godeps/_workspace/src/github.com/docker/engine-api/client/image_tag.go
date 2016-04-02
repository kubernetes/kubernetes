package client

import (
	"net/url"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// ImageTag tags an image in the docker host
func (cli *Client) ImageTag(ctx context.Context, options types.ImageTagOptions) error {
	query := url.Values{}
	query.Set("repo", options.RepositoryName)
	query.Set("tag", options.Tag)
	if options.Force {
		query.Set("force", "1")
	}

	resp, err := cli.post(ctx, "/images/"+options.ImageID+"/tag", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
