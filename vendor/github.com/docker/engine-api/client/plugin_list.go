// +build experimental

package client

import (
	"encoding/json"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// PluginList returns the installed plugins
func (cli *Client) PluginList(ctx context.Context) (types.PluginsListResponse, error) {
	var plugins types.PluginsListResponse
	resp, err := cli.get(ctx, "/plugins", nil, nil)
	if err != nil {
		return plugins, err
	}

	err = json.NewDecoder(resp.body).Decode(&plugins)
	ensureReaderClosed(resp)
	return plugins, err
}
