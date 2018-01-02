// +build !linux

package plugin

import (
	"errors"
	"io"
	"net/http"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"golang.org/x/net/context"
)

var errNotSupported = errors.New("plugins are not supported on this platform")

// Disable deactivates a plugin, which implies that they cannot be used by containers.
func (pm *Manager) Disable(name string, config *types.PluginDisableConfig) error {
	return errNotSupported
}

// Enable activates a plugin, which implies that they are ready to be used by containers.
func (pm *Manager) Enable(name string, config *types.PluginEnableConfig) error {
	return errNotSupported
}

// Inspect examines a plugin config
func (pm *Manager) Inspect(refOrID string) (tp *types.Plugin, err error) {
	return nil, errNotSupported
}

// Privileges pulls a plugin config and computes the privileges required to install it.
func (pm *Manager) Privileges(ctx context.Context, ref reference.Named, metaHeader http.Header, authConfig *types.AuthConfig) (types.PluginPrivileges, error) {
	return nil, errNotSupported
}

// Pull pulls a plugin, check if the correct privileges are provided and install the plugin.
func (pm *Manager) Pull(ctx context.Context, ref reference.Named, name string, metaHeader http.Header, authConfig *types.AuthConfig, privileges types.PluginPrivileges, out io.Writer, opts ...CreateOpt) error {
	return errNotSupported
}

// Upgrade pulls a plugin, check if the correct privileges are provided and install the plugin.
func (pm *Manager) Upgrade(ctx context.Context, ref reference.Named, name string, metaHeader http.Header, authConfig *types.AuthConfig, privileges types.PluginPrivileges, outStream io.Writer) error {
	return errNotSupported
}

// List displays the list of plugins and associated metadata.
func (pm *Manager) List(pluginFilters filters.Args) ([]types.Plugin, error) {
	return nil, errNotSupported
}

// Push pushes a plugin to the store.
func (pm *Manager) Push(ctx context.Context, name string, metaHeader http.Header, authConfig *types.AuthConfig, out io.Writer) error {
	return errNotSupported
}

// Remove deletes plugin's root directory.
func (pm *Manager) Remove(name string, config *types.PluginRmConfig) error {
	return errNotSupported
}

// Set sets plugin args
func (pm *Manager) Set(name string, args []string) error {
	return errNotSupported
}

// CreateFromContext creates a plugin from the given pluginDir which contains
// both the rootfs and the config.json and a repoName with optional tag.
func (pm *Manager) CreateFromContext(ctx context.Context, tarCtx io.ReadCloser, options *types.PluginCreateOptions) error {
	return errNotSupported
}
