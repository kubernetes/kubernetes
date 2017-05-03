package client

const (
	RECREATE_ON_QUORUM_STRATEGY_CONFIG_TYPE = "recreateOnQuorumStrategyConfig"
)

type RecreateOnQuorumStrategyConfig struct {
	Resource

	Quorum int64 `json:"quorum,omitempty" yaml:"quorum,omitempty"`
}

type RecreateOnQuorumStrategyConfigCollection struct {
	Collection
	Data []RecreateOnQuorumStrategyConfig `json:"data,omitempty"`
}

type RecreateOnQuorumStrategyConfigClient struct {
	rancherClient *RancherClient
}

type RecreateOnQuorumStrategyConfigOperations interface {
	List(opts *ListOpts) (*RecreateOnQuorumStrategyConfigCollection, error)
	Create(opts *RecreateOnQuorumStrategyConfig) (*RecreateOnQuorumStrategyConfig, error)
	Update(existing *RecreateOnQuorumStrategyConfig, updates interface{}) (*RecreateOnQuorumStrategyConfig, error)
	ById(id string) (*RecreateOnQuorumStrategyConfig, error)
	Delete(container *RecreateOnQuorumStrategyConfig) error
}

func newRecreateOnQuorumStrategyConfigClient(rancherClient *RancherClient) *RecreateOnQuorumStrategyConfigClient {
	return &RecreateOnQuorumStrategyConfigClient{
		rancherClient: rancherClient,
	}
}

func (c *RecreateOnQuorumStrategyConfigClient) Create(container *RecreateOnQuorumStrategyConfig) (*RecreateOnQuorumStrategyConfig, error) {
	resp := &RecreateOnQuorumStrategyConfig{}
	err := c.rancherClient.doCreate(RECREATE_ON_QUORUM_STRATEGY_CONFIG_TYPE, container, resp)
	return resp, err
}

func (c *RecreateOnQuorumStrategyConfigClient) Update(existing *RecreateOnQuorumStrategyConfig, updates interface{}) (*RecreateOnQuorumStrategyConfig, error) {
	resp := &RecreateOnQuorumStrategyConfig{}
	err := c.rancherClient.doUpdate(RECREATE_ON_QUORUM_STRATEGY_CONFIG_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *RecreateOnQuorumStrategyConfigClient) List(opts *ListOpts) (*RecreateOnQuorumStrategyConfigCollection, error) {
	resp := &RecreateOnQuorumStrategyConfigCollection{}
	err := c.rancherClient.doList(RECREATE_ON_QUORUM_STRATEGY_CONFIG_TYPE, opts, resp)
	return resp, err
}

func (c *RecreateOnQuorumStrategyConfigClient) ById(id string) (*RecreateOnQuorumStrategyConfig, error) {
	resp := &RecreateOnQuorumStrategyConfig{}
	err := c.rancherClient.doById(RECREATE_ON_QUORUM_STRATEGY_CONFIG_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *RecreateOnQuorumStrategyConfigClient) Delete(container *RecreateOnQuorumStrategyConfig) error {
	return c.rancherClient.doResourceDelete(RECREATE_ON_QUORUM_STRATEGY_CONFIG_TYPE, &container.Resource)
}
