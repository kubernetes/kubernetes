// +build experimental

package client

import (
	"golang.org/x/net/context"
)

// PluginEnable enables a plugin
func (cli *Client) PluginEnable(ctx context.Context, name string) error {
	resp, err := cli.post(ctx, "/plugins/"+name+"/enable", nil, nil, nil)
	ensureReaderClosed(resp)
	return err
}
