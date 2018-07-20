package swarm

import (
	basictypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	types "github.com/docker/docker/api/types/swarm"
	"golang.org/x/net/context"
)

// Backend abstracts a swarm manager.
type Backend interface {
	Init(req types.InitRequest) (string, error)
	Join(req types.JoinRequest) error
	Leave(force bool) error
	Inspect() (types.Swarm, error)
	Update(uint64, types.Spec, types.UpdateFlags) error
	GetUnlockKey() (string, error)
	UnlockSwarm(req types.UnlockRequest) error

	GetServices(basictypes.ServiceListOptions) ([]types.Service, error)
	GetService(idOrName string, insertDefaults bool) (types.Service, error)
	CreateService(types.ServiceSpec, string, bool) (*basictypes.ServiceCreateResponse, error)
	UpdateService(string, uint64, types.ServiceSpec, basictypes.ServiceUpdateOptions, bool) (*basictypes.ServiceUpdateResponse, error)
	RemoveService(string) error

	ServiceLogs(context.Context, *backend.LogSelector, *basictypes.ContainerLogsOptions) (<-chan *backend.LogMessage, error)

	GetNodes(basictypes.NodeListOptions) ([]types.Node, error)
	GetNode(string) (types.Node, error)
	UpdateNode(string, uint64, types.NodeSpec) error
	RemoveNode(string, bool) error

	GetTasks(basictypes.TaskListOptions) ([]types.Task, error)
	GetTask(string) (types.Task, error)

	GetSecrets(opts basictypes.SecretListOptions) ([]types.Secret, error)
	CreateSecret(s types.SecretSpec) (string, error)
	RemoveSecret(idOrName string) error
	GetSecret(id string) (types.Secret, error)
	UpdateSecret(idOrName string, version uint64, spec types.SecretSpec) error

	GetConfigs(opts basictypes.ConfigListOptions) ([]types.Config, error)
	CreateConfig(s types.ConfigSpec) (string, error)
	RemoveConfig(id string) error
	GetConfig(id string) (types.Config, error)
	UpdateConfig(idOrName string, version uint64, spec types.ConfigSpec) error
}
