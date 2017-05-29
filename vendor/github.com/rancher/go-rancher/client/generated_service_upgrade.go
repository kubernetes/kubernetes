package client

const (
	SERVICE_UPGRADE_TYPE = "serviceUpgrade"
)

type ServiceUpgrade struct {
	Resource

	InServiceStrategy *InServiceUpgradeStrategy `json:"inServiceStrategy,omitempty" yaml:"in_service_strategy,omitempty"`

	ToServiceStrategy *ToServiceUpgradeStrategy `json:"toServiceStrategy,omitempty" yaml:"to_service_strategy,omitempty"`
}

type ServiceUpgradeCollection struct {
	Collection
	Data []ServiceUpgrade `json:"data,omitempty"`
}

type ServiceUpgradeClient struct {
	rancherClient *RancherClient
}

type ServiceUpgradeOperations interface {
	List(opts *ListOpts) (*ServiceUpgradeCollection, error)
	Create(opts *ServiceUpgrade) (*ServiceUpgrade, error)
	Update(existing *ServiceUpgrade, updates interface{}) (*ServiceUpgrade, error)
	ById(id string) (*ServiceUpgrade, error)
	Delete(container *ServiceUpgrade) error
}

func newServiceUpgradeClient(rancherClient *RancherClient) *ServiceUpgradeClient {
	return &ServiceUpgradeClient{
		rancherClient: rancherClient,
	}
}

func (c *ServiceUpgradeClient) Create(container *ServiceUpgrade) (*ServiceUpgrade, error) {
	resp := &ServiceUpgrade{}
	err := c.rancherClient.doCreate(SERVICE_UPGRADE_TYPE, container, resp)
	return resp, err
}

func (c *ServiceUpgradeClient) Update(existing *ServiceUpgrade, updates interface{}) (*ServiceUpgrade, error) {
	resp := &ServiceUpgrade{}
	err := c.rancherClient.doUpdate(SERVICE_UPGRADE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ServiceUpgradeClient) List(opts *ListOpts) (*ServiceUpgradeCollection, error) {
	resp := &ServiceUpgradeCollection{}
	err := c.rancherClient.doList(SERVICE_UPGRADE_TYPE, opts, resp)
	return resp, err
}

func (c *ServiceUpgradeClient) ById(id string) (*ServiceUpgrade, error) {
	resp := &ServiceUpgrade{}
	err := c.rancherClient.doById(SERVICE_UPGRADE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ServiceUpgradeClient) Delete(container *ServiceUpgrade) error {
	return c.rancherClient.doResourceDelete(SERVICE_UPGRADE_TYPE, &container.Resource)
}
