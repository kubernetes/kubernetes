package client

const (
	TO_SERVICE_UPGRADE_STRATEGY_TYPE = "toServiceUpgradeStrategy"
)

type ToServiceUpgradeStrategy struct {
	Resource

	BatchSize int64 `json:"batchSize,omitempty" yaml:"batch_size,omitempty"`

	FinalScale int64 `json:"finalScale,omitempty" yaml:"final_scale,omitempty"`

	IntervalMillis int64 `json:"intervalMillis,omitempty" yaml:"interval_millis,omitempty"`

	ToServiceId string `json:"toServiceId,omitempty" yaml:"to_service_id,omitempty"`

	UpdateLinks bool `json:"updateLinks,omitempty" yaml:"update_links,omitempty"`
}

type ToServiceUpgradeStrategyCollection struct {
	Collection
	Data []ToServiceUpgradeStrategy `json:"data,omitempty"`
}

type ToServiceUpgradeStrategyClient struct {
	rancherClient *RancherClient
}

type ToServiceUpgradeStrategyOperations interface {
	List(opts *ListOpts) (*ToServiceUpgradeStrategyCollection, error)
	Create(opts *ToServiceUpgradeStrategy) (*ToServiceUpgradeStrategy, error)
	Update(existing *ToServiceUpgradeStrategy, updates interface{}) (*ToServiceUpgradeStrategy, error)
	ById(id string) (*ToServiceUpgradeStrategy, error)
	Delete(container *ToServiceUpgradeStrategy) error
}

func newToServiceUpgradeStrategyClient(rancherClient *RancherClient) *ToServiceUpgradeStrategyClient {
	return &ToServiceUpgradeStrategyClient{
		rancherClient: rancherClient,
	}
}

func (c *ToServiceUpgradeStrategyClient) Create(container *ToServiceUpgradeStrategy) (*ToServiceUpgradeStrategy, error) {
	resp := &ToServiceUpgradeStrategy{}
	err := c.rancherClient.doCreate(TO_SERVICE_UPGRADE_STRATEGY_TYPE, container, resp)
	return resp, err
}

func (c *ToServiceUpgradeStrategyClient) Update(existing *ToServiceUpgradeStrategy, updates interface{}) (*ToServiceUpgradeStrategy, error) {
	resp := &ToServiceUpgradeStrategy{}
	err := c.rancherClient.doUpdate(TO_SERVICE_UPGRADE_STRATEGY_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ToServiceUpgradeStrategyClient) List(opts *ListOpts) (*ToServiceUpgradeStrategyCollection, error) {
	resp := &ToServiceUpgradeStrategyCollection{}
	err := c.rancherClient.doList(TO_SERVICE_UPGRADE_STRATEGY_TYPE, opts, resp)
	return resp, err
}

func (c *ToServiceUpgradeStrategyClient) ById(id string) (*ToServiceUpgradeStrategy, error) {
	resp := &ToServiceUpgradeStrategy{}
	err := c.rancherClient.doById(TO_SERVICE_UPGRADE_STRATEGY_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ToServiceUpgradeStrategyClient) Delete(container *ToServiceUpgradeStrategy) error {
	return c.rancherClient.doResourceDelete(TO_SERVICE_UPGRADE_STRATEGY_TYPE, &container.Resource)
}
