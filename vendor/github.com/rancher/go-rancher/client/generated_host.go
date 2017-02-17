package client

const (
	HOST_TYPE = "host"
)

type Host struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	AgentId string `json:"agentId,omitempty" yaml:"agent_id,omitempty"`

	AgentState string `json:"agentState,omitempty" yaml:"agent_state,omitempty"`

	ApiProxy string `json:"apiProxy,omitempty" yaml:"api_proxy,omitempty"`

	ComputeTotal int64 `json:"computeTotal,omitempty" yaml:"compute_total,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Hostname string `json:"hostname,omitempty" yaml:"hostname,omitempty"`

	Info interface{} `json:"info,omitempty" yaml:"info,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Labels map[string]interface{} `json:"labels,omitempty" yaml:"labels,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	PhysicalHostId string `json:"physicalHostId,omitempty" yaml:"physical_host_id,omitempty"`

	PublicEndpoints []interface{} `json:"publicEndpoints,omitempty" yaml:"public_endpoints,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type HostCollection struct {
	Collection
	Data []Host `json:"data,omitempty"`
}

type HostClient struct {
	rancherClient *RancherClient
}

type HostOperations interface {
	List(opts *ListOpts) (*HostCollection, error)
	Create(opts *Host) (*Host, error)
	Update(existing *Host, updates interface{}) (*Host, error)
	ById(id string) (*Host, error)
	Delete(container *Host) error

	ActionActivate(*Host) (*Host, error)

	ActionCreate(*Host) (*Host, error)

	ActionDeactivate(*Host) (*Host, error)

	ActionDockersocket(*Host) (*HostAccess, error)

	ActionPurge(*Host) (*Host, error)

	ActionRemove(*Host) (*Host, error)

	ActionRestore(*Host) (*Host, error)

	ActionUpdate(*Host) (*Host, error)
}

func newHostClient(rancherClient *RancherClient) *HostClient {
	return &HostClient{
		rancherClient: rancherClient,
	}
}

func (c *HostClient) Create(container *Host) (*Host, error) {
	resp := &Host{}
	err := c.rancherClient.doCreate(HOST_TYPE, container, resp)
	return resp, err
}

func (c *HostClient) Update(existing *Host, updates interface{}) (*Host, error) {
	resp := &Host{}
	err := c.rancherClient.doUpdate(HOST_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *HostClient) List(opts *ListOpts) (*HostCollection, error) {
	resp := &HostCollection{}
	err := c.rancherClient.doList(HOST_TYPE, opts, resp)
	return resp, err
}

func (c *HostClient) ById(id string) (*Host, error) {
	resp := &Host{}
	err := c.rancherClient.doById(HOST_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *HostClient) Delete(container *Host) error {
	return c.rancherClient.doResourceDelete(HOST_TYPE, &container.Resource)
}

func (c *HostClient) ActionActivate(resource *Host) (*Host, error) {

	resp := &Host{}

	err := c.rancherClient.doAction(HOST_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HostClient) ActionCreate(resource *Host) (*Host, error) {

	resp := &Host{}

	err := c.rancherClient.doAction(HOST_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HostClient) ActionDeactivate(resource *Host) (*Host, error) {

	resp := &Host{}

	err := c.rancherClient.doAction(HOST_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HostClient) ActionDockersocket(resource *Host) (*HostAccess, error) {

	resp := &HostAccess{}

	err := c.rancherClient.doAction(HOST_TYPE, "dockersocket", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HostClient) ActionPurge(resource *Host) (*Host, error) {

	resp := &Host{}

	err := c.rancherClient.doAction(HOST_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HostClient) ActionRemove(resource *Host) (*Host, error) {

	resp := &Host{}

	err := c.rancherClient.doAction(HOST_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HostClient) ActionRestore(resource *Host) (*Host, error) {

	resp := &Host{}

	err := c.rancherClient.doAction(HOST_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HostClient) ActionUpdate(resource *Host) (*Host, error) {

	resp := &Host{}

	err := c.rancherClient.doAction(HOST_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
