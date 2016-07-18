// +build experimental

package client

import (
	"golang.org/x/net/context"
)

// PluginPush pushes a plugin to a registry
func (cli *Client) PluginPush(ctx context.Context, name string, registryAuth string) error {
	headers := map[string][]string{"X-Registry-Auth": {registryAuth}}
	resp, err := cli.post(ctx, "/plugins/"+name+"/push", nil, nil, headers)
	ensureReaderClosed(resp)
	return err
}
