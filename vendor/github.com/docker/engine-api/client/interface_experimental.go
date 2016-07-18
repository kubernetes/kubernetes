// +build experimental

package client

import (
	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// APIClient is an interface that clients that talk with a docker server must implement.
type APIClient interface {
	CommonAPIClient
	CheckpointAPIClient
	PluginAPIClient
}

// CheckpointAPIClient defines API client methods for the checkpoints
type CheckpointAPIClient interface {
	CheckpointCreate(ctx context.Context, container string, options types.CheckpointCreateOptions) error
	CheckpointDelete(ctx context.Context, container string, checkpointID string) error
	CheckpointList(ctx context.Context, container string) ([]types.Checkpoint, error)
}

// PluginAPIClient defines API client methods for the plugins
type PluginAPIClient interface {
	PluginList(ctx context.Context) (types.PluginsListResponse, error)
	PluginRemove(ctx context.Context, name string) error
	PluginEnable(ctx context.Context, name string) error
	PluginDisable(ctx context.Context, name string) error
	PluginInstall(ctx context.Context, name string, options types.PluginInstallOptions) error
	PluginPush(ctx context.Context, name string, registryAuth string) error
	PluginSet(ctx context.Context, name string, args []string) error
	PluginInspect(ctx context.Context, name string) (*types.Plugin, error)
}

// Ensure that Client always implements APIClient.
var _ APIClient = &Client{}
