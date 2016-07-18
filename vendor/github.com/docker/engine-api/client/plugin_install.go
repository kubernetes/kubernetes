// +build experimental

package client

import (
	"encoding/json"
	"net/http"
	"net/url"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// PluginInstall installs a plugin
func (cli *Client) PluginInstall(ctx context.Context, name string, options types.PluginInstallOptions) error {
	// FIXME(vdemeester) name is a ref, we might want to parse/validate it here.
	query := url.Values{}
	query.Set("name", name)
	resp, err := cli.tryPluginPull(ctx, query, options.RegistryAuth)
	if resp.statusCode == http.StatusUnauthorized && options.PrivilegeFunc != nil {
		newAuthHeader, privilegeErr := options.PrivilegeFunc()
		if privilegeErr != nil {
			ensureReaderClosed(resp)
			return privilegeErr
		}
		resp, err = cli.tryPluginPull(ctx, query, newAuthHeader)
	}
	if err != nil {
		ensureReaderClosed(resp)
		return err
	}
	var privileges types.PluginPrivileges
	if err := json.NewDecoder(resp.body).Decode(&privileges); err != nil {
		ensureReaderClosed(resp)
		return err
	}
	ensureReaderClosed(resp)

	if !options.AcceptAllPermissions && options.AcceptPermissionsFunc != nil && len(privileges) > 0 {
		accept, err := options.AcceptPermissionsFunc(privileges)
		if err != nil {
			return err
		}
		if !accept {
			resp, _ := cli.delete(ctx, "/plugins/"+name, nil, nil)
			ensureReaderClosed(resp)
			return pluginPermissionDenied{name}
		}
	}
	if options.Disabled {
		return nil
	}
	return cli.PluginEnable(ctx, name)
}

func (cli *Client) tryPluginPull(ctx context.Context, query url.Values, registryAuth string) (*serverResponse, error) {
	headers := map[string][]string{"X-Registry-Auth": {registryAuth}}
	return cli.post(ctx, "/plugins/pull", query, nil, headers)
}
