package client

const (
	CONTAINER_EVENT_TYPE = "containerEvent"
)

type ContainerEvent struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	DockerInspect interface{} `json:"dockerInspect,omitempty" yaml:"docker_inspect,omitempty"`

	ExternalFrom string `json:"externalFrom,omitempty" yaml:"external_from,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	ExternalStatus string `json:"externalStatus,omitempty" yaml:"external_status,omitempty"`

	ExternalTimestamp int64 `json:"externalTimestamp,omitempty" yaml:"external_timestamp,omitempty"`

	HostId string `json:"hostId,omitempty" yaml:"host_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	ReportedHostUuid string `json:"reportedHostUuid,omitempty" yaml:"reported_host_uuid,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`
}

type ContainerEventCollection struct {
	Collection
	Data []ContainerEvent `json:"data,omitempty"`
}

type ContainerEventClient struct {
	rancherClient *RancherClient
}

type ContainerEventOperations interface {
	List(opts *ListOpts) (*ContainerEventCollection, error)
	Create(opts *ContainerEvent) (*ContainerEvent, error)
	Update(existing *ContainerEvent, updates interface{}) (*ContainerEvent, error)
	ById(id string) (*ContainerEvent, error)
	Delete(container *ContainerEvent) error

	ActionCreate(*ContainerEvent) (*ContainerEvent, error)

	ActionRemove(*ContainerEvent) (*ContainerEvent, error)
}

func newContainerEventClient(rancherClient *RancherClient) *ContainerEventClient {
	return &ContainerEventClient{
		rancherClient: rancherClient,
	}
}

func (c *ContainerEventClient) Create(container *ContainerEvent) (*ContainerEvent, error) {
	resp := &ContainerEvent{}
	err := c.rancherClient.doCreate(CONTAINER_EVENT_TYPE, container, resp)
	return resp, err
}

func (c *ContainerEventClient) Update(existing *ContainerEvent, updates interface{}) (*ContainerEvent, error) {
	resp := &ContainerEvent{}
	err := c.rancherClient.doUpdate(CONTAINER_EVENT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ContainerEventClient) List(opts *ListOpts) (*ContainerEventCollection, error) {
	resp := &ContainerEventCollection{}
	err := c.rancherClient.doList(CONTAINER_EVENT_TYPE, opts, resp)
	return resp, err
}

func (c *ContainerEventClient) ById(id string) (*ContainerEvent, error) {
	resp := &ContainerEvent{}
	err := c.rancherClient.doById(CONTAINER_EVENT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ContainerEventClient) Delete(container *ContainerEvent) error {
	return c.rancherClient.doResourceDelete(CONTAINER_EVENT_TYPE, &container.Resource)
}

func (c *ContainerEventClient) ActionCreate(resource *ContainerEvent) (*ContainerEvent, error) {

	resp := &ContainerEvent{}

	err := c.rancherClient.doAction(CONTAINER_EVENT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ContainerEventClient) ActionRemove(resource *ContainerEvent) (*ContainerEvent, error) {

	resp := &ContainerEvent{}

	err := c.rancherClient.doAction(CONTAINER_EVENT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
