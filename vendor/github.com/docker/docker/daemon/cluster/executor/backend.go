package executor

import (
	"io"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/events"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/network"
	swarmtypes "github.com/docker/docker/api/types/swarm"
	containerpkg "github.com/docker/docker/container"
	clustertypes "github.com/docker/docker/daemon/cluster/provider"
	"github.com/docker/docker/plugin"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/cluster"
	networktypes "github.com/docker/libnetwork/types"
	"github.com/docker/swarmkit/agent/exec"
	"golang.org/x/net/context"
)

// Backend defines the executor component for a swarm agent.
type Backend interface {
	CreateManagedNetwork(clustertypes.NetworkCreateRequest) error
	DeleteManagedNetwork(name string) error
	FindNetwork(idName string) (libnetwork.Network, error)
	SetupIngress(clustertypes.NetworkCreateRequest, string) (<-chan struct{}, error)
	ReleaseIngress() (<-chan struct{}, error)
	PullImage(ctx context.Context, image, tag, platform string, metaHeaders map[string][]string, authConfig *types.AuthConfig, outStream io.Writer) error
	CreateManagedContainer(config types.ContainerCreateConfig) (container.ContainerCreateCreatedBody, error)
	ContainerStart(name string, hostConfig *container.HostConfig, checkpoint string, checkpointDir string) error
	ContainerStop(name string, seconds *int) error
	ContainerLogs(context.Context, string, *types.ContainerLogsOptions) (<-chan *backend.LogMessage, error)
	ConnectContainerToNetwork(containerName, networkName string, endpointConfig *network.EndpointSettings) error
	ActivateContainerServiceBinding(containerName string) error
	DeactivateContainerServiceBinding(containerName string) error
	UpdateContainerServiceConfig(containerName string, serviceConfig *clustertypes.ServiceConfig) error
	ContainerInspectCurrent(name string, size bool) (*types.ContainerJSON, error)
	ContainerWait(ctx context.Context, name string, condition containerpkg.WaitCondition) (<-chan containerpkg.StateStatus, error)
	ContainerRm(name string, config *types.ContainerRmConfig) error
	ContainerKill(name string, sig uint64) error
	SetContainerDependencyStore(name string, store exec.DependencyGetter) error
	SetContainerSecretReferences(name string, refs []*swarmtypes.SecretReference) error
	SetContainerConfigReferences(name string, refs []*swarmtypes.ConfigReference) error
	SystemInfo() (*types.Info, error)
	VolumeCreate(name, driverName string, opts, labels map[string]string) (*types.Volume, error)
	Containers(config *types.ContainerListOptions) ([]*types.Container, error)
	SetNetworkBootstrapKeys([]*networktypes.EncryptionKey) error
	DaemonJoinsCluster(provider cluster.Provider)
	DaemonLeavesCluster()
	IsSwarmCompatible() error
	SubscribeToEvents(since, until time.Time, filter filters.Args) ([]events.Message, chan interface{})
	UnsubscribeFromEvents(listener chan interface{})
	UpdateAttachment(string, string, string, *network.NetworkingConfig) error
	WaitForDetachment(context.Context, string, string, string, string) error
	GetRepository(context.Context, reference.Named, *types.AuthConfig) (distribution.Repository, bool, error)
	LookupImage(name string) (*types.ImageInspect, error)
	PluginManager() *plugin.Manager
	PluginGetter() *plugin.Store
}
