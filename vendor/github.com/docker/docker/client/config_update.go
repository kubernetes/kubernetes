package client // import "github.com/docker/docker/client"

import (
	"context"
	"net/url"
	"strconv"

	"github.com/docker/docker/api/types/swarm"
)

// ConfigUpdate attempts to update a Config
func (cli *Client) ConfigUpdate(ctx context.Context, id string, version swarm.Version, config swarm.ConfigSpec) error {
	if err := cli.NewVersionError("1.30", "config update"); err != nil {
		return err
	}
	query := url.Values{}
	query.Set("version", strconv.FormatUint(version.Index, 10))
	resp, err := cli.post(ctx, "/configs/"+id+"/update", query, config, nil)
	ensureReaderClosed(resp)
	return err
}
