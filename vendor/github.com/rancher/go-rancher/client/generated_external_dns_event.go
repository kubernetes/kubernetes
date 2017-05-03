package client

const (
	EXTERNAL_DNS_EVENT_TYPE = "externalDnsEvent"
)

type ExternalDnsEvent struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	EventType string `json:"eventType,omitempty" yaml:"event_type,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	Fqdn string `json:"fqdn,omitempty" yaml:"fqdn,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	ReportedAccountId string `json:"reportedAccountId,omitempty" yaml:"reported_account_id,omitempty"`

	ServiceName string `json:"serviceName,omitempty" yaml:"service_name,omitempty"`

	StackName string `json:"stackName,omitempty" yaml:"stack_name,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type ExternalDnsEventCollection struct {
	Collection
	Data []ExternalDnsEvent `json:"data,omitempty"`
}

type ExternalDnsEventClient struct {
	rancherClient *RancherClient
}

type ExternalDnsEventOperations interface {
	List(opts *ListOpts) (*ExternalDnsEventCollection, error)
	Create(opts *ExternalDnsEvent) (*ExternalDnsEvent, error)
	Update(existing *ExternalDnsEvent, updates interface{}) (*ExternalDnsEvent, error)
	ById(id string) (*ExternalDnsEvent, error)
	Delete(container *ExternalDnsEvent) error

	ActionCreate(*ExternalDnsEvent) (*ExternalEvent, error)

	ActionRemove(*ExternalDnsEvent) (*ExternalEvent, error)
}

func newExternalDnsEventClient(rancherClient *RancherClient) *ExternalDnsEventClient {
	return &ExternalDnsEventClient{
		rancherClient: rancherClient,
	}
}

func (c *ExternalDnsEventClient) Create(container *ExternalDnsEvent) (*ExternalDnsEvent, error) {
	resp := &ExternalDnsEvent{}
	err := c.rancherClient.doCreate(EXTERNAL_DNS_EVENT_TYPE, container, resp)
	return resp, err
}

func (c *ExternalDnsEventClient) Update(existing *ExternalDnsEvent, updates interface{}) (*ExternalDnsEvent, error) {
	resp := &ExternalDnsEvent{}
	err := c.rancherClient.doUpdate(EXTERNAL_DNS_EVENT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ExternalDnsEventClient) List(opts *ListOpts) (*ExternalDnsEventCollection, error) {
	resp := &ExternalDnsEventCollection{}
	err := c.rancherClient.doList(EXTERNAL_DNS_EVENT_TYPE, opts, resp)
	return resp, err
}

func (c *ExternalDnsEventClient) ById(id string) (*ExternalDnsEvent, error) {
	resp := &ExternalDnsEvent{}
	err := c.rancherClient.doById(EXTERNAL_DNS_EVENT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ExternalDnsEventClient) Delete(container *ExternalDnsEvent) error {
	return c.rancherClient.doResourceDelete(EXTERNAL_DNS_EVENT_TYPE, &container.Resource)
}

func (c *ExternalDnsEventClient) ActionCreate(resource *ExternalDnsEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_DNS_EVENT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalDnsEventClient) ActionRemove(resource *ExternalDnsEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_DNS_EVENT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
