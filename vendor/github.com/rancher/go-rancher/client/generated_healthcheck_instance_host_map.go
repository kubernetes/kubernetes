package client

const (
	HEALTHCHECK_INSTANCE_HOST_MAP_TYPE = "healthcheckInstanceHostMap"
)

type HealthcheckInstanceHostMap struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	HealthState string `json:"healthState,omitempty" yaml:"health_state,omitempty"`

	HostId string `json:"hostId,omitempty" yaml:"host_id,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

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

type HealthcheckInstanceHostMapCollection struct {
	Collection
	Data []HealthcheckInstanceHostMap `json:"data,omitempty"`
}

type HealthcheckInstanceHostMapClient struct {
	rancherClient *RancherClient
}

type HealthcheckInstanceHostMapOperations interface {
	List(opts *ListOpts) (*HealthcheckInstanceHostMapCollection, error)
	Create(opts *HealthcheckInstanceHostMap) (*HealthcheckInstanceHostMap, error)
	Update(existing *HealthcheckInstanceHostMap, updates interface{}) (*HealthcheckInstanceHostMap, error)
	ById(id string) (*HealthcheckInstanceHostMap, error)
	Delete(container *HealthcheckInstanceHostMap) error

	ActionCreate(*HealthcheckInstanceHostMap) (*HealthcheckInstanceHostMap, error)

	ActionRemove(*HealthcheckInstanceHostMap) (*HealthcheckInstanceHostMap, error)
}

func newHealthcheckInstanceHostMapClient(rancherClient *RancherClient) *HealthcheckInstanceHostMapClient {
	return &HealthcheckInstanceHostMapClient{
		rancherClient: rancherClient,
	}
}

func (c *HealthcheckInstanceHostMapClient) Create(container *HealthcheckInstanceHostMap) (*HealthcheckInstanceHostMap, error) {
	resp := &HealthcheckInstanceHostMap{}
	err := c.rancherClient.doCreate(HEALTHCHECK_INSTANCE_HOST_MAP_TYPE, container, resp)
	return resp, err
}

func (c *HealthcheckInstanceHostMapClient) Update(existing *HealthcheckInstanceHostMap, updates interface{}) (*HealthcheckInstanceHostMap, error) {
	resp := &HealthcheckInstanceHostMap{}
	err := c.rancherClient.doUpdate(HEALTHCHECK_INSTANCE_HOST_MAP_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *HealthcheckInstanceHostMapClient) List(opts *ListOpts) (*HealthcheckInstanceHostMapCollection, error) {
	resp := &HealthcheckInstanceHostMapCollection{}
	err := c.rancherClient.doList(HEALTHCHECK_INSTANCE_HOST_MAP_TYPE, opts, resp)
	return resp, err
}

func (c *HealthcheckInstanceHostMapClient) ById(id string) (*HealthcheckInstanceHostMap, error) {
	resp := &HealthcheckInstanceHostMap{}
	err := c.rancherClient.doById(HEALTHCHECK_INSTANCE_HOST_MAP_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *HealthcheckInstanceHostMapClient) Delete(container *HealthcheckInstanceHostMap) error {
	return c.rancherClient.doResourceDelete(HEALTHCHECK_INSTANCE_HOST_MAP_TYPE, &container.Resource)
}

func (c *HealthcheckInstanceHostMapClient) ActionCreate(resource *HealthcheckInstanceHostMap) (*HealthcheckInstanceHostMap, error) {

	resp := &HealthcheckInstanceHostMap{}

	err := c.rancherClient.doAction(HEALTHCHECK_INSTANCE_HOST_MAP_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *HealthcheckInstanceHostMapClient) ActionRemove(resource *HealthcheckInstanceHostMap) (*HealthcheckInstanceHostMap, error) {

	resp := &HealthcheckInstanceHostMap{}

	err := c.rancherClient.doAction(HEALTHCHECK_INSTANCE_HOST_MAP_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
