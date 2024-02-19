package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"net/url"
	"path"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/api/types/versions"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

type configWrapper struct {
	*container.Config
	HostConfig       *container.HostConfig
	NetworkingConfig *network.NetworkingConfig
}

// ContainerCreate creates a new container based in the given configuration.
// It can be associated with a name, but it's not mandatory.
func (cli *Client) ContainerCreate(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, platform *specs.Platform, containerName string) (container.ContainerCreateCreatedBody, error) {
	var response container.ContainerCreateCreatedBody

	if err := cli.NewVersionError("1.25", "stop timeout"); config != nil && config.StopTimeout != nil && err != nil {
		return response, err
	}

	// When using API 1.24 and under, the client is responsible for removing the container
	if hostConfig != nil && versions.LessThan(cli.ClientVersion(), "1.25") {
		hostConfig.AutoRemove = false
	}

	if err := cli.NewVersionError("1.41", "specify container image platform"); platform != nil && err != nil {
		return response, err
	}

	query := url.Values{}
	if p := formatPlatform(platform); p != "" {
		query.Set("platform", p)
	}

	if containerName != "" {
		query.Set("name", containerName)
	}

	body := configWrapper{
		Config:           config,
		HostConfig:       hostConfig,
		NetworkingConfig: networkingConfig,
	}

	serverResp, err := cli.post(ctx, "/containers/create", query, body, nil)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return response, err
	}

	err = json.NewDecoder(serverResp.body).Decode(&response)
	return response, err
}

// formatPlatform returns a formatted string representing platform (e.g. linux/arm/v7).
//
// Similar to containerd's platforms.Format(), but does allow components to be
// omitted (e.g. pass "architecture" only, without "os":
// https://github.com/containerd/containerd/blob/v1.5.2/platforms/platforms.go#L243-L263
func formatPlatform(platform *specs.Platform) string {
	if platform == nil {
		return ""
	}
	return path.Join(platform.OS, platform.Architecture, platform.Variant)
}
