package client

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

// PluginInstall installs a plugin
func (cli *Client) PluginInstall(ctx context.Context, name string, options types.PluginInstallOptions) (rc io.ReadCloser, err error) {
	query := url.Values{}
	if _, err := reference.ParseNamed(options.RemoteRef); err != nil {
		return nil, errors.Wrap(err, "invalid remote reference")
	}
	query.Set("remote", options.RemoteRef)

	privileges, err := cli.checkPluginPermissions(ctx, query, options)
	if err != nil {
		return nil, err
	}

	// set name for plugin pull, if empty should default to remote reference
	query.Set("name", name)

	resp, err := cli.tryPluginPull(ctx, query, privileges, options.RegistryAuth)
	if err != nil {
		return nil, err
	}

	name = resp.header.Get("Docker-Plugin-Name")

	pr, pw := io.Pipe()
	go func() { // todo: the client should probably be designed more around the actual api
		_, err := io.Copy(pw, resp.body)
		if err != nil {
			pw.CloseWithError(err)
			return
		}
		defer func() {
			if err != nil {
				delResp, _ := cli.delete(ctx, "/plugins/"+name, nil, nil)
				ensureReaderClosed(delResp)
			}
		}()
		if len(options.Args) > 0 {
			if err := cli.PluginSet(ctx, name, options.Args); err != nil {
				pw.CloseWithError(err)
				return
			}
		}

		if options.Disabled {
			pw.Close()
			return
		}

		err = cli.PluginEnable(ctx, name, types.PluginEnableOptions{Timeout: 0})
		pw.CloseWithError(err)
	}()
	return pr, nil
}

func (cli *Client) tryPluginPrivileges(ctx context.Context, query url.Values, registryAuth string) (serverResponse, error) {
	headers := map[string][]string{"X-Registry-Auth": {registryAuth}}
	return cli.get(ctx, "/plugins/privileges", query, headers)
}

func (cli *Client) tryPluginPull(ctx context.Context, query url.Values, privileges types.PluginPrivileges, registryAuth string) (serverResponse, error) {
	headers := map[string][]string{"X-Registry-Auth": {registryAuth}}
	return cli.post(ctx, "/plugins/pull", query, privileges, headers)
}

func (cli *Client) checkPluginPermissions(ctx context.Context, query url.Values, options types.PluginInstallOptions) (types.PluginPrivileges, error) {
	resp, err := cli.tryPluginPrivileges(ctx, query, options.RegistryAuth)
	if resp.statusCode == http.StatusUnauthorized && options.PrivilegeFunc != nil {
		// todo: do inspect before to check existing name before checking privileges
		newAuthHeader, privilegeErr := options.PrivilegeFunc()
		if privilegeErr != nil {
			ensureReaderClosed(resp)
			return nil, privilegeErr
		}
		options.RegistryAuth = newAuthHeader
		resp, err = cli.tryPluginPrivileges(ctx, query, options.RegistryAuth)
	}
	if err != nil {
		ensureReaderClosed(resp)
		return nil, err
	}

	var privileges types.PluginPrivileges
	if err := json.NewDecoder(resp.body).Decode(&privileges); err != nil {
		ensureReaderClosed(resp)
		return nil, err
	}
	ensureReaderClosed(resp)

	if !options.AcceptAllPermissions && options.AcceptPermissionsFunc != nil && len(privileges) > 0 {
		accept, err := options.AcceptPermissionsFunc(privileges)
		if err != nil {
			return nil, err
		}
		if !accept {
			return nil, pluginPermissionDenied{options.RemoteRef}
		}
	}
	return privileges, nil
}
