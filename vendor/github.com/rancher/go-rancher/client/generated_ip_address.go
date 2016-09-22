package client

const (
	IP_ADDRESS_TYPE = "ipAddress"
)

type IpAddress struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Address string `json:"address,omitempty" yaml:"address,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	NetworkId string `json:"networkId,omitempty" yaml:"network_id,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type IpAddressCollection struct {
	Collection
	Data []IpAddress `json:"data,omitempty"`
}

type IpAddressClient struct {
	rancherClient *RancherClient
}

type IpAddressOperations interface {
	List(opts *ListOpts) (*IpAddressCollection, error)
	Create(opts *IpAddress) (*IpAddress, error)
	Update(existing *IpAddress, updates interface{}) (*IpAddress, error)
	ById(id string) (*IpAddress, error)
	Delete(container *IpAddress) error

	ActionActivate(*IpAddress) (*IpAddress, error)

	ActionCreate(*IpAddress) (*IpAddress, error)

	ActionDeactivate(*IpAddress) (*IpAddress, error)

	ActionDisassociate(*IpAddress) (*IpAddress, error)

	ActionPurge(*IpAddress) (*IpAddress, error)

	ActionRemove(*IpAddress) (*IpAddress, error)

	ActionRestore(*IpAddress) (*IpAddress, error)

	ActionUpdate(*IpAddress) (*IpAddress, error)
}

func newIpAddressClient(rancherClient *RancherClient) *IpAddressClient {
	return &IpAddressClient{
		rancherClient: rancherClient,
	}
}

func (c *IpAddressClient) Create(container *IpAddress) (*IpAddress, error) {
	resp := &IpAddress{}
	err := c.rancherClient.doCreate(IP_ADDRESS_TYPE, container, resp)
	return resp, err
}

func (c *IpAddressClient) Update(existing *IpAddress, updates interface{}) (*IpAddress, error) {
	resp := &IpAddress{}
	err := c.rancherClient.doUpdate(IP_ADDRESS_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *IpAddressClient) List(opts *ListOpts) (*IpAddressCollection, error) {
	resp := &IpAddressCollection{}
	err := c.rancherClient.doList(IP_ADDRESS_TYPE, opts, resp)
	return resp, err
}

func (c *IpAddressClient) ById(id string) (*IpAddress, error) {
	resp := &IpAddress{}
	err := c.rancherClient.doById(IP_ADDRESS_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *IpAddressClient) Delete(container *IpAddress) error {
	return c.rancherClient.doResourceDelete(IP_ADDRESS_TYPE, &container.Resource)
}

func (c *IpAddressClient) ActionActivate(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *IpAddressClient) ActionCreate(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *IpAddressClient) ActionDeactivate(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *IpAddressClient) ActionDisassociate(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "disassociate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *IpAddressClient) ActionPurge(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *IpAddressClient) ActionRemove(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *IpAddressClient) ActionRestore(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *IpAddressClient) ActionUpdate(resource *IpAddress) (*IpAddress, error) {

	resp := &IpAddress{}

	err := c.rancherClient.doAction(IP_ADDRESS_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
