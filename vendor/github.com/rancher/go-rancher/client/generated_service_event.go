package client

const (
	SERVICE_EVENT_TYPE = "serviceEvent"
)

type ServiceEvent struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	ExternalTimestamp int64 `json:"externalTimestamp,omitempty" yaml:"external_timestamp,omitempty"`

	HealthcheckUuid string `json:"healthcheckUuid,omitempty" yaml:"healthcheck_uuid,omitempty"`

	HostId string `json:"hostId,omitempty" yaml:"host_id,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	ReportedHealth string `json:"reportedHealth,omitempty" yaml:"reported_health,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type ServiceEventCollection struct {
	Collection
	Data []ServiceEvent `json:"data,omitempty"`
}

type ServiceEventClient struct {
	rancherClient *RancherClient
}

type ServiceEventOperations interface {
	List(opts *ListOpts) (*ServiceEventCollection, error)
	Create(opts *ServiceEvent) (*ServiceEvent, error)
	Update(existing *ServiceEvent, updates interface{}) (*ServiceEvent, error)
	ById(id string) (*ServiceEvent, error)
	Delete(container *ServiceEvent) error

	ActionCreate(*ServiceEvent) (*ServiceEvent, error)

	ActionRemove(*ServiceEvent) (*ServiceEvent, error)
}

func newServiceEventClient(rancherClient *RancherClient) *ServiceEventClient {
	return &ServiceEventClient{
		rancherClient: rancherClient,
	}
}

func (c *ServiceEventClient) Create(container *ServiceEvent) (*ServiceEvent, error) {
	resp := &ServiceEvent{}
	err := c.rancherClient.doCreate(SERVICE_EVENT_TYPE, container, resp)
	return resp, err
}

func (c *ServiceEventClient) Update(existing *ServiceEvent, updates interface{}) (*ServiceEvent, error) {
	resp := &ServiceEvent{}
	err := c.rancherClient.doUpdate(SERVICE_EVENT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ServiceEventClient) List(opts *ListOpts) (*ServiceEventCollection, error) {
	resp := &ServiceEventCollection{}
	err := c.rancherClient.doList(SERVICE_EVENT_TYPE, opts, resp)
	return resp, err
}

func (c *ServiceEventClient) ById(id string) (*ServiceEvent, error) {
	resp := &ServiceEvent{}
	err := c.rancherClient.doById(SERVICE_EVENT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ServiceEventClient) Delete(container *ServiceEvent) error {
	return c.rancherClient.doResourceDelete(SERVICE_EVENT_TYPE, &container.Resource)
}

func (c *ServiceEventClient) ActionCreate(resource *ServiceEvent) (*ServiceEvent, error) {

	resp := &ServiceEvent{}

	err := c.rancherClient.doAction(SERVICE_EVENT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ServiceEventClient) ActionRemove(resource *ServiceEvent) (*ServiceEvent, error) {

	resp := &ServiceEvent{}

	err := c.rancherClient.doAction(SERVICE_EVENT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
