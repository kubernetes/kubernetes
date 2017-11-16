package client

const (
	PORT_TYPE = "port"
)

type Port struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	BindAddress string `json:"bindAddress,omitempty" yaml:"bind_address,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	PrivateIpAddressId string `json:"privateIpAddressId,omitempty" yaml:"private_ip_address_id,omitempty"`

	PrivatePort int64 `json:"privatePort,omitempty" yaml:"private_port,omitempty"`

	Protocol string `json:"protocol,omitempty" yaml:"protocol,omitempty"`

	PublicIpAddressId string `json:"publicIpAddressId,omitempty" yaml:"public_ip_address_id,omitempty"`

	PublicPort int64 `json:"publicPort,omitempty" yaml:"public_port,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type PortCollection struct {
	Collection
	Data []Port `json:"data,omitempty"`
}

type PortClient struct {
	rancherClient *RancherClient
}

type PortOperations interface {
	List(opts *ListOpts) (*PortCollection, error)
	Create(opts *Port) (*Port, error)
	Update(existing *Port, updates interface{}) (*Port, error)
	ById(id string) (*Port, error)
	Delete(container *Port) error

	ActionActivate(*Port) (*Port, error)

	ActionCreate(*Port) (*Port, error)

	ActionDeactivate(*Port) (*Port, error)

	ActionPurge(*Port) (*Port, error)

	ActionRemove(*Port) (*Port, error)

	ActionRestore(*Port) (*Port, error)

	ActionUpdate(*Port) (*Port, error)
}

func newPortClient(rancherClient *RancherClient) *PortClient {
	return &PortClient{
		rancherClient: rancherClient,
	}
}

func (c *PortClient) Create(container *Port) (*Port, error) {
	resp := &Port{}
	err := c.rancherClient.doCreate(PORT_TYPE, container, resp)
	return resp, err
}

func (c *PortClient) Update(existing *Port, updates interface{}) (*Port, error) {
	resp := &Port{}
	err := c.rancherClient.doUpdate(PORT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *PortClient) List(opts *ListOpts) (*PortCollection, error) {
	resp := &PortCollection{}
	err := c.rancherClient.doList(PORT_TYPE, opts, resp)
	return resp, err
}

func (c *PortClient) ById(id string) (*Port, error) {
	resp := &Port{}
	err := c.rancherClient.doById(PORT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *PortClient) Delete(container *Port) error {
	return c.rancherClient.doResourceDelete(PORT_TYPE, &container.Resource)
}

func (c *PortClient) ActionActivate(resource *Port) (*Port, error) {

	resp := &Port{}

	err := c.rancherClient.doAction(PORT_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PortClient) ActionCreate(resource *Port) (*Port, error) {

	resp := &Port{}

	err := c.rancherClient.doAction(PORT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PortClient) ActionDeactivate(resource *Port) (*Port, error) {

	resp := &Port{}

	err := c.rancherClient.doAction(PORT_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PortClient) ActionPurge(resource *Port) (*Port, error) {

	resp := &Port{}

	err := c.rancherClient.doAction(PORT_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PortClient) ActionRemove(resource *Port) (*Port, error) {

	resp := &Port{}

	err := c.rancherClient.doAction(PORT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PortClient) ActionRestore(resource *Port) (*Port, error) {

	resp := &Port{}

	err := c.rancherClient.doAction(PORT_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PortClient) ActionUpdate(resource *Port) (*Port, error) {

	resp := &Port{}

	err := c.rancherClient.doAction(PORT_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
