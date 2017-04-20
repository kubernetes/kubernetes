package client

const (
	EXTERNAL_SERVICE_EVENT_TYPE = "externalServiceEvent"
)

type ExternalServiceEvent struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Environment interface{} `json:"environment,omitempty" yaml:"environment,omitempty"`

	EventType string `json:"eventType,omitempty" yaml:"event_type,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	ReportedAccountId string `json:"reportedAccountId,omitempty" yaml:"reported_account_id,omitempty"`

	Service interface{} `json:"service,omitempty" yaml:"service,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type ExternalServiceEventCollection struct {
	Collection
	Data []ExternalServiceEvent `json:"data,omitempty"`
}

type ExternalServiceEventClient struct {
	rancherClient *RancherClient
}

type ExternalServiceEventOperations interface {
	List(opts *ListOpts) (*ExternalServiceEventCollection, error)
	Create(opts *ExternalServiceEvent) (*ExternalServiceEvent, error)
	Update(existing *ExternalServiceEvent, updates interface{}) (*ExternalServiceEvent, error)
	ById(id string) (*ExternalServiceEvent, error)
	Delete(container *ExternalServiceEvent) error

	ActionCreate(*ExternalServiceEvent) (*ExternalEvent, error)

	ActionRemove(*ExternalServiceEvent) (*ExternalEvent, error)
}

func newExternalServiceEventClient(rancherClient *RancherClient) *ExternalServiceEventClient {
	return &ExternalServiceEventClient{
		rancherClient: rancherClient,
	}
}

func (c *ExternalServiceEventClient) Create(container *ExternalServiceEvent) (*ExternalServiceEvent, error) {
	resp := &ExternalServiceEvent{}
	err := c.rancherClient.doCreate(EXTERNAL_SERVICE_EVENT_TYPE, container, resp)
	return resp, err
}

func (c *ExternalServiceEventClient) Update(existing *ExternalServiceEvent, updates interface{}) (*ExternalServiceEvent, error) {
	resp := &ExternalServiceEvent{}
	err := c.rancherClient.doUpdate(EXTERNAL_SERVICE_EVENT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ExternalServiceEventClient) List(opts *ListOpts) (*ExternalServiceEventCollection, error) {
	resp := &ExternalServiceEventCollection{}
	err := c.rancherClient.doList(EXTERNAL_SERVICE_EVENT_TYPE, opts, resp)
	return resp, err
}

func (c *ExternalServiceEventClient) ById(id string) (*ExternalServiceEvent, error) {
	resp := &ExternalServiceEvent{}
	err := c.rancherClient.doById(EXTERNAL_SERVICE_EVENT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ExternalServiceEventClient) Delete(container *ExternalServiceEvent) error {
	return c.rancherClient.doResourceDelete(EXTERNAL_SERVICE_EVENT_TYPE, &container.Resource)
}

func (c *ExternalServiceEventClient) ActionCreate(resource *ExternalServiceEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_EVENT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalServiceEventClient) ActionRemove(resource *ExternalServiceEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_SERVICE_EVENT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
