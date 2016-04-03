package client

import (
	"io"

	"golang.org/x/net/context"

	"github.com/docker/engine-api/types"
	"github.com/docker/engine-api/types/container"
	"github.com/docker/engine-api/types/filters"
	"github.com/docker/engine-api/types/network"
	"github.com/docker/engine-api/types/registry"
)

// APIClient is an interface that clients that talk with a docker server must implement.
type APIClient interface {
	ClientVersion() string
	ContainerAttach(ctx context.Context, options types.ContainerAttachOptions) (types.HijackedResponse, error)
	ContainerCommit(ctx context.Context, options types.ContainerCommitOptions) (types.ContainerCommitResponse, error)
	ContainerCreate(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, containerName string) (types.ContainerCreateResponse, error)
	ContainerDiff(ctx context.Context, ontainerID string) ([]types.ContainerChange, error)
	ContainerExecAttach(ctx context.Context, execID string, config types.ExecConfig) (types.HijackedResponse, error)
	ContainerExecCreate(ctx context.Context, config types.ExecConfig) (types.ContainerExecCreateResponse, error)
	ContainerExecInspect(ctx context.Context, execID string) (types.ContainerExecInspect, error)
	ContainerExecResize(ctx context.Context, options types.ResizeOptions) error
	ContainerExecStart(ctx context.Context, execID string, config types.ExecStartCheck) error
	ContainerExport(ctx context.Context, containerID string) (io.ReadCloser, error)
	ContainerInspect(ctx context.Context, containerID string) (types.ContainerJSON, error)
	ContainerInspectWithRaw(ctx context.Context, containerID string, getSize bool) (types.ContainerJSON, []byte, error)
	ContainerKill(ctx context.Context, containerID, signal string) error
	ContainerList(ctx context.Context, options types.ContainerListOptions) ([]types.Container, error)
	ContainerLogs(ctx context.Context, options types.ContainerLogsOptions) (io.ReadCloser, error)
	ContainerPause(ctx context.Context, containerID string) error
	ContainerRemove(ctx context.Context, options types.ContainerRemoveOptions) error
	ContainerRename(ctx context.Context, containerID, newContainerName string) error
	ContainerResize(ctx context.Context, options types.ResizeOptions) error
	ContainerRestart(ctx context.Context, containerID string, timeout int) error
	ContainerStatPath(ctx context.Context, containerID, path string) (types.ContainerPathStat, error)
	ContainerStats(ctx context.Context, containerID string, stream bool) (io.ReadCloser, error)
	ContainerStart(ctx context.Context, containerID string) error
	ContainerStop(ctx context.Context, containerID string, timeout int) error
	ContainerTop(ctx context.Context, containerID string, arguments []string) (types.ContainerProcessList, error)
	ContainerUnpause(ctx context.Context, containerID string) error
	ContainerUpdate(ctx context.Context, containerID string, updateConfig container.UpdateConfig) error
	ContainerWait(ctx context.Context, containerID string) (int, error)
	CopyFromContainer(ctx context.Context, containerID, srcPath string) (io.ReadCloser, types.ContainerPathStat, error)
	CopyToContainer(ctx context.Context, options types.CopyToContainerOptions) error
	Events(ctx context.Context, options types.EventsOptions) (io.ReadCloser, error)
	ImageBuild(ctx context.Context, options types.ImageBuildOptions) (types.ImageBuildResponse, error)
	ImageCreate(ctx context.Context, options types.ImageCreateOptions) (io.ReadCloser, error)
	ImageHistory(ctx context.Context, imageID string) ([]types.ImageHistory, error)
	ImageImport(ctx context.Context, options types.ImageImportOptions) (io.ReadCloser, error)
	ImageInspectWithRaw(ctx context.Context, imageID string, getSize bool) (types.ImageInspect, []byte, error)
	ImageList(ctx context.Context, options types.ImageListOptions) ([]types.Image, error)
	ImageLoad(ctx context.Context, input io.Reader, quiet bool) (types.ImageLoadResponse, error)
	ImagePull(ctx context.Context, options types.ImagePullOptions, privilegeFunc RequestPrivilegeFunc) (io.ReadCloser, error)
	ImagePush(ctx context.Context, options types.ImagePushOptions, privilegeFunc RequestPrivilegeFunc) (io.ReadCloser, error)
	ImageRemove(ctx context.Context, options types.ImageRemoveOptions) ([]types.ImageDelete, error)
	ImageSearch(ctx context.Context, options types.ImageSearchOptions, privilegeFunc RequestPrivilegeFunc) ([]registry.SearchResult, error)
	ImageSave(ctx context.Context, imageIDs []string) (io.ReadCloser, error)
	ImageTag(ctx context.Context, options types.ImageTagOptions) error
	Info(ctx context.Context) (types.Info, error)
	NetworkConnect(ctx context.Context, networkID, containerID string, config *network.EndpointSettings) error
	NetworkCreate(ctx context.Context, options types.NetworkCreate) (types.NetworkCreateResponse, error)
	NetworkDisconnect(ctx context.Context, networkID, containerID string, force bool) error
	NetworkInspect(ctx context.Context, networkID string) (types.NetworkResource, error)
	NetworkList(ctx context.Context, options types.NetworkListOptions) ([]types.NetworkResource, error)
	NetworkRemove(ctx context.Context, networkID string) error
	RegistryLogin(ctx context.Context, auth types.AuthConfig) (types.AuthResponse, error)
	ServerVersion(ctx context.Context) (types.Version, error)
	VolumeCreate(ctx context.Context, options types.VolumeCreateRequest) (types.Volume, error)
	VolumeInspect(ctx context.Context, volumeID string) (types.Volume, error)
	VolumeList(ctx context.Context, filter filters.Args) (types.VolumesListResponse, error)
	VolumeRemove(ctx context.Context, volumeID string) error
}

// Ensure that Client always implements APIClient.
var _ APIClient = &Client{}
