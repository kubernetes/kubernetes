// +build experimental

package client

import (
	"golang.org/x/net/context"
)

// PluginRemove removes a plugin
func (cli *Client) PluginRemove(ctx context.Context, name string) error {
	resp, err := cli.delete(ctx, "/plugins/"+name, nil, nil)
	ensureReaderClosed(resp)
	return err
}
