package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"net/url"

	"github.com/docker/docker/api/types"
)

// ImageRemove removes an image from the docker host.
func (cli *Client) ImageRemove(ctx context.Context, imageID string, options types.ImageRemoveOptions) ([]types.ImageDeleteResponseItem, error) {
	query := url.Values{}

	if options.Force {
		query.Set("force", "1")
	}
	if !options.PruneChildren {
		query.Set("noprune", "1")
	}

	var dels []types.ImageDeleteResponseItem
	resp, err := cli.delete(ctx, "/images/"+imageID, query, nil)
	defer ensureReaderClosed(resp)
	if err != nil {
		return dels, wrapResponseError(err, resp, "image", imageID)
	}

	err = json.NewDecoder(resp.body).Decode(&dels)
	return dels, err
}
