package client

const (
	INSTANCE_LINK_TYPE = "instanceLink"
)

type InstanceLink struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	LinkName string `json:"linkName,omitempty" yaml:"link_name,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Ports []interface{} `json:"ports,omitempty" yaml:"ports,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	TargetInstanceId string `json:"targetInstanceId,omitempty" yaml:"target_instance_id,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type InstanceLinkCollection struct {
	Collection
	Data []InstanceLink `json:"data,omitempty"`
}

type InstanceLinkClient struct {
	rancherClient *RancherClient
}

type InstanceLinkOperations interface {
	List(opts *ListOpts) (*InstanceLinkCollection, error)
	Create(opts *InstanceLink) (*InstanceLink, error)
	Update(existing *InstanceLink, updates interface{}) (*InstanceLink, error)
	ById(id string) (*InstanceLink, error)
	Delete(container *InstanceLink) error

	ActionActivate(*InstanceLink) (*InstanceLink, error)

	ActionCreate(*InstanceLink) (*InstanceLink, error)

	ActionDeactivate(*InstanceLink) (*InstanceLink, error)

	ActionPurge(*InstanceLink) (*InstanceLink, error)

	ActionRemove(*InstanceLink) (*InstanceLink, error)

	ActionRestore(*InstanceLink) (*InstanceLink, error)

	ActionUpdate(*InstanceLink) (*InstanceLink, error)
}

func newInstanceLinkClient(rancherClient *RancherClient) *InstanceLinkClient {
	return &InstanceLinkClient{
		rancherClient: rancherClient,
	}
}

func (c *InstanceLinkClient) Create(container *InstanceLink) (*InstanceLink, error) {
	resp := &InstanceLink{}
	err := c.rancherClient.doCreate(INSTANCE_LINK_TYPE, container, resp)
	return resp, err
}

func (c *InstanceLinkClient) Update(existing *InstanceLink, updates interface{}) (*InstanceLink, error) {
	resp := &InstanceLink{}
	err := c.rancherClient.doUpdate(INSTANCE_LINK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *InstanceLinkClient) List(opts *ListOpts) (*InstanceLinkCollection, error) {
	resp := &InstanceLinkCollection{}
	err := c.rancherClient.doList(INSTANCE_LINK_TYPE, opts, resp)
	return resp, err
}

func (c *InstanceLinkClient) ById(id string) (*InstanceLink, error) {
	resp := &InstanceLink{}
	err := c.rancherClient.doById(INSTANCE_LINK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *InstanceLinkClient) Delete(container *InstanceLink) error {
	return c.rancherClient.doResourceDelete(INSTANCE_LINK_TYPE, &container.Resource)
}

func (c *InstanceLinkClient) ActionActivate(resource *InstanceLink) (*InstanceLink, error) {

	resp := &InstanceLink{}

	err := c.rancherClient.doAction(INSTANCE_LINK_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *InstanceLinkClient) ActionCreate(resource *InstanceLink) (*InstanceLink, error) {

	resp := &InstanceLink{}

	err := c.rancherClient.doAction(INSTANCE_LINK_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *InstanceLinkClient) ActionDeactivate(resource *InstanceLink) (*InstanceLink, error) {

	resp := &InstanceLink{}

	err := c.rancherClient.doAction(INSTANCE_LINK_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *InstanceLinkClient) ActionPurge(resource *InstanceLink) (*InstanceLink, error) {

	resp := &InstanceLink{}

	err := c.rancherClient.doAction(INSTANCE_LINK_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *InstanceLinkClient) ActionRemove(resource *InstanceLink) (*InstanceLink, error) {

	resp := &InstanceLink{}

	err := c.rancherClient.doAction(INSTANCE_LINK_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *InstanceLinkClient) ActionRestore(resource *InstanceLink) (*InstanceLink, error) {

	resp := &InstanceLink{}

	err := c.rancherClient.doAction(INSTANCE_LINK_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *InstanceLinkClient) ActionUpdate(resource *InstanceLink) (*InstanceLink, error) {

	resp := &InstanceLink{}

	err := c.rancherClient.doAction(INSTANCE_LINK_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
