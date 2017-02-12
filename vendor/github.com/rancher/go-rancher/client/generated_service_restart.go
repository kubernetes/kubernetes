package client

const (
	SERVICE_RESTART_TYPE = "serviceRestart"
)

type ServiceRestart struct {
	Resource

	RollingRestartStrategy RollingRestartStrategy `json:"rollingRestartStrategy,omitempty" yaml:"rolling_restart_strategy,omitempty"`
}

type ServiceRestartCollection struct {
	Collection
	Data []ServiceRestart `json:"data,omitempty"`
}

type ServiceRestartClient struct {
	rancherClient *RancherClient
}

type ServiceRestartOperations interface {
	List(opts *ListOpts) (*ServiceRestartCollection, error)
	Create(opts *ServiceRestart) (*ServiceRestart, error)
	Update(existing *ServiceRestart, updates interface{}) (*ServiceRestart, error)
	ById(id string) (*ServiceRestart, error)
	Delete(container *ServiceRestart) error
}

func newServiceRestartClient(rancherClient *RancherClient) *ServiceRestartClient {
	return &ServiceRestartClient{
		rancherClient: rancherClient,
	}
}

func (c *ServiceRestartClient) Create(container *ServiceRestart) (*ServiceRestart, error) {
	resp := &ServiceRestart{}
	err := c.rancherClient.doCreate(SERVICE_RESTART_TYPE, container, resp)
	return resp, err
}

func (c *ServiceRestartClient) Update(existing *ServiceRestart, updates interface{}) (*ServiceRestart, error) {
	resp := &ServiceRestart{}
	err := c.rancherClient.doUpdate(SERVICE_RESTART_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ServiceRestartClient) List(opts *ListOpts) (*ServiceRestartCollection, error) {
	resp := &ServiceRestartCollection{}
	err := c.rancherClient.doList(SERVICE_RESTART_TYPE, opts, resp)
	return resp, err
}

func (c *ServiceRestartClient) ById(id string) (*ServiceRestart, error) {
	resp := &ServiceRestart{}
	err := c.rancherClient.doById(SERVICE_RESTART_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ServiceRestartClient) Delete(container *ServiceRestart) error {
	return c.rancherClient.doResourceDelete(SERVICE_RESTART_TYPE, &container.Resource)
}
