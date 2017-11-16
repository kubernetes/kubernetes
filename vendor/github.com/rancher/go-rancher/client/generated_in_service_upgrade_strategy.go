package client

const (
	IN_SERVICE_UPGRADE_STRATEGY_TYPE = "inServiceUpgradeStrategy"
)

type InServiceUpgradeStrategy struct {
	Resource

	BatchSize int64 `json:"batchSize,omitempty" yaml:"batch_size,omitempty"`

	IntervalMillis int64 `json:"intervalMillis,omitempty" yaml:"interval_millis,omitempty"`

	LaunchConfig *LaunchConfig `json:"launchConfig,omitempty" yaml:"launch_config,omitempty"`

	PreviousLaunchConfig *LaunchConfig `json:"previousLaunchConfig,omitempty" yaml:"previous_launch_config,omitempty"`

	PreviousSecondaryLaunchConfigs []interface{} `json:"previousSecondaryLaunchConfigs,omitempty" yaml:"previous_secondary_launch_configs,omitempty"`

	SecondaryLaunchConfigs []interface{} `json:"secondaryLaunchConfigs,omitempty" yaml:"secondary_launch_configs,omitempty"`

	StartFirst bool `json:"startFirst,omitempty" yaml:"start_first,omitempty"`
}

type InServiceUpgradeStrategyCollection struct {
	Collection
	Data []InServiceUpgradeStrategy `json:"data,omitempty"`
}

type InServiceUpgradeStrategyClient struct {
	rancherClient *RancherClient
}

type InServiceUpgradeStrategyOperations interface {
	List(opts *ListOpts) (*InServiceUpgradeStrategyCollection, error)
	Create(opts *InServiceUpgradeStrategy) (*InServiceUpgradeStrategy, error)
	Update(existing *InServiceUpgradeStrategy, updates interface{}) (*InServiceUpgradeStrategy, error)
	ById(id string) (*InServiceUpgradeStrategy, error)
	Delete(container *InServiceUpgradeStrategy) error
}

func newInServiceUpgradeStrategyClient(rancherClient *RancherClient) *InServiceUpgradeStrategyClient {
	return &InServiceUpgradeStrategyClient{
		rancherClient: rancherClient,
	}
}

func (c *InServiceUpgradeStrategyClient) Create(container *InServiceUpgradeStrategy) (*InServiceUpgradeStrategy, error) {
	resp := &InServiceUpgradeStrategy{}
	err := c.rancherClient.doCreate(IN_SERVICE_UPGRADE_STRATEGY_TYPE, container, resp)
	return resp, err
}

func (c *InServiceUpgradeStrategyClient) Update(existing *InServiceUpgradeStrategy, updates interface{}) (*InServiceUpgradeStrategy, error) {
	resp := &InServiceUpgradeStrategy{}
	err := c.rancherClient.doUpdate(IN_SERVICE_UPGRADE_STRATEGY_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *InServiceUpgradeStrategyClient) List(opts *ListOpts) (*InServiceUpgradeStrategyCollection, error) {
	resp := &InServiceUpgradeStrategyCollection{}
	err := c.rancherClient.doList(IN_SERVICE_UPGRADE_STRATEGY_TYPE, opts, resp)
	return resp, err
}

func (c *InServiceUpgradeStrategyClient) ById(id string) (*InServiceUpgradeStrategy, error) {
	resp := &InServiceUpgradeStrategy{}
	err := c.rancherClient.doById(IN_SERVICE_UPGRADE_STRATEGY_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *InServiceUpgradeStrategyClient) Delete(container *InServiceUpgradeStrategy) error {
	return c.rancherClient.doResourceDelete(IN_SERVICE_UPGRADE_STRATEGY_TYPE, &container.Resource)
}
