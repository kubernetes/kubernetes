package client

const (
	EXTERNAL_HOST_EVENT_TYPE = "externalHostEvent"
)

type ExternalHostEvent struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	DeleteHost bool `json:"deleteHost,omitempty" yaml:"delete_host,omitempty"`

	EventType string `json:"eventType,omitempty" yaml:"event_type,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	HostId string `json:"hostId,omitempty" yaml:"host_id,omitempty"`

	HostLabel string `json:"hostLabel,omitempty" yaml:"host_label,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	ReportedAccountId string `json:"reportedAccountId,omitempty" yaml:"reported_account_id,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type ExternalHostEventCollection struct {
	Collection
	Data []ExternalHostEvent `json:"data,omitempty"`
}

type ExternalHostEventClient struct {
	rancherClient *RancherClient
}

type ExternalHostEventOperations interface {
	List(opts *ListOpts) (*ExternalHostEventCollection, error)
	Create(opts *ExternalHostEvent) (*ExternalHostEvent, error)
	Update(existing *ExternalHostEvent, updates interface{}) (*ExternalHostEvent, error)
	ById(id string) (*ExternalHostEvent, error)
	Delete(container *ExternalHostEvent) error

	ActionCreate(*ExternalHostEvent) (*ExternalEvent, error)

	ActionRemove(*ExternalHostEvent) (*ExternalEvent, error)
}

func newExternalHostEventClient(rancherClient *RancherClient) *ExternalHostEventClient {
	return &ExternalHostEventClient{
		rancherClient: rancherClient,
	}
}

func (c *ExternalHostEventClient) Create(container *ExternalHostEvent) (*ExternalHostEvent, error) {
	resp := &ExternalHostEvent{}
	err := c.rancherClient.doCreate(EXTERNAL_HOST_EVENT_TYPE, container, resp)
	return resp, err
}

func (c *ExternalHostEventClient) Update(existing *ExternalHostEvent, updates interface{}) (*ExternalHostEvent, error) {
	resp := &ExternalHostEvent{}
	err := c.rancherClient.doUpdate(EXTERNAL_HOST_EVENT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ExternalHostEventClient) List(opts *ListOpts) (*ExternalHostEventCollection, error) {
	resp := &ExternalHostEventCollection{}
	err := c.rancherClient.doList(EXTERNAL_HOST_EVENT_TYPE, opts, resp)
	return resp, err
}

func (c *ExternalHostEventClient) ById(id string) (*ExternalHostEvent, error) {
	resp := &ExternalHostEvent{}
	err := c.rancherClient.doById(EXTERNAL_HOST_EVENT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ExternalHostEventClient) Delete(container *ExternalHostEvent) error {
	return c.rancherClient.doResourceDelete(EXTERNAL_HOST_EVENT_TYPE, &container.Resource)
}

func (c *ExternalHostEventClient) ActionCreate(resource *ExternalHostEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_HOST_EVENT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalHostEventClient) ActionRemove(resource *ExternalHostEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_HOST_EVENT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
