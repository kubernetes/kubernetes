package client

const (
	PHYSICAL_HOST_TYPE = "physicalHost"
)

type PhysicalHost struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Driver string `json:"driver,omitempty" yaml:"driver,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type PhysicalHostCollection struct {
	Collection
	Data []PhysicalHost `json:"data,omitempty"`
}

type PhysicalHostClient struct {
	rancherClient *RancherClient
}

type PhysicalHostOperations interface {
	List(opts *ListOpts) (*PhysicalHostCollection, error)
	Create(opts *PhysicalHost) (*PhysicalHost, error)
	Update(existing *PhysicalHost, updates interface{}) (*PhysicalHost, error)
	ById(id string) (*PhysicalHost, error)
	Delete(container *PhysicalHost) error

	ActionBootstrap(*PhysicalHost) (*PhysicalHost, error)

	ActionCreate(*PhysicalHost) (*PhysicalHost, error)

	ActionError(*PhysicalHost) (*PhysicalHost, error)

	ActionRemove(*PhysicalHost) (*PhysicalHost, error)

	ActionUpdate(*PhysicalHost) (*PhysicalHost, error)
}

func newPhysicalHostClient(rancherClient *RancherClient) *PhysicalHostClient {
	return &PhysicalHostClient{
		rancherClient: rancherClient,
	}
}

func (c *PhysicalHostClient) Create(container *PhysicalHost) (*PhysicalHost, error) {
	resp := &PhysicalHost{}
	err := c.rancherClient.doCreate(PHYSICAL_HOST_TYPE, container, resp)
	return resp, err
}

func (c *PhysicalHostClient) Update(existing *PhysicalHost, updates interface{}) (*PhysicalHost, error) {
	resp := &PhysicalHost{}
	err := c.rancherClient.doUpdate(PHYSICAL_HOST_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *PhysicalHostClient) List(opts *ListOpts) (*PhysicalHostCollection, error) {
	resp := &PhysicalHostCollection{}
	err := c.rancherClient.doList(PHYSICAL_HOST_TYPE, opts, resp)
	return resp, err
}

func (c *PhysicalHostClient) ById(id string) (*PhysicalHost, error) {
	resp := &PhysicalHost{}
	err := c.rancherClient.doById(PHYSICAL_HOST_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *PhysicalHostClient) Delete(container *PhysicalHost) error {
	return c.rancherClient.doResourceDelete(PHYSICAL_HOST_TYPE, &container.Resource)
}

func (c *PhysicalHostClient) ActionBootstrap(resource *PhysicalHost) (*PhysicalHost, error) {

	resp := &PhysicalHost{}

	err := c.rancherClient.doAction(PHYSICAL_HOST_TYPE, "bootstrap", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PhysicalHostClient) ActionCreate(resource *PhysicalHost) (*PhysicalHost, error) {

	resp := &PhysicalHost{}

	err := c.rancherClient.doAction(PHYSICAL_HOST_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PhysicalHostClient) ActionError(resource *PhysicalHost) (*PhysicalHost, error) {

	resp := &PhysicalHost{}

	err := c.rancherClient.doAction(PHYSICAL_HOST_TYPE, "error", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PhysicalHostClient) ActionRemove(resource *PhysicalHost) (*PhysicalHost, error) {

	resp := &PhysicalHost{}

	err := c.rancherClient.doAction(PHYSICAL_HOST_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PhysicalHostClient) ActionUpdate(resource *PhysicalHost) (*PhysicalHost, error) {

	resp := &PhysicalHost{}

	err := c.rancherClient.doAction(PHYSICAL_HOST_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
