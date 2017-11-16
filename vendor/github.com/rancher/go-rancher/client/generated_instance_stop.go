package client

const (
	INSTANCE_STOP_TYPE = "instanceStop"
)

type InstanceStop struct {
	Resource

	Remove bool `json:"remove,omitempty" yaml:"remove,omitempty"`

	Timeout int64 `json:"timeout,omitempty" yaml:"timeout,omitempty"`
}

type InstanceStopCollection struct {
	Collection
	Data []InstanceStop `json:"data,omitempty"`
}

type InstanceStopClient struct {
	rancherClient *RancherClient
}

type InstanceStopOperations interface {
	List(opts *ListOpts) (*InstanceStopCollection, error)
	Create(opts *InstanceStop) (*InstanceStop, error)
	Update(existing *InstanceStop, updates interface{}) (*InstanceStop, error)
	ById(id string) (*InstanceStop, error)
	Delete(container *InstanceStop) error
}

func newInstanceStopClient(rancherClient *RancherClient) *InstanceStopClient {
	return &InstanceStopClient{
		rancherClient: rancherClient,
	}
}

func (c *InstanceStopClient) Create(container *InstanceStop) (*InstanceStop, error) {
	resp := &InstanceStop{}
	err := c.rancherClient.doCreate(INSTANCE_STOP_TYPE, container, resp)
	return resp, err
}

func (c *InstanceStopClient) Update(existing *InstanceStop, updates interface{}) (*InstanceStop, error) {
	resp := &InstanceStop{}
	err := c.rancherClient.doUpdate(INSTANCE_STOP_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *InstanceStopClient) List(opts *ListOpts) (*InstanceStopCollection, error) {
	resp := &InstanceStopCollection{}
	err := c.rancherClient.doList(INSTANCE_STOP_TYPE, opts, resp)
	return resp, err
}

func (c *InstanceStopClient) ById(id string) (*InstanceStop, error) {
	resp := &InstanceStop{}
	err := c.rancherClient.doById(INSTANCE_STOP_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *InstanceStopClient) Delete(container *InstanceStop) error {
	return c.rancherClient.doResourceDelete(INSTANCE_STOP_TYPE, &container.Resource)
}
