package client

const (
	EXTERNAL_VOLUME_EVENT_TYPE = "externalVolumeEvent"
)

type ExternalVolumeEvent struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	EventType string `json:"eventType,omitempty" yaml:"event_type,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	ReportedAccountId string `json:"reportedAccountId,omitempty" yaml:"reported_account_id,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`

	Volume Volume `json:"volume,omitempty" yaml:"volume,omitempty"`
}

type ExternalVolumeEventCollection struct {
	Collection
	Data []ExternalVolumeEvent `json:"data,omitempty"`
}

type ExternalVolumeEventClient struct {
	rancherClient *RancherClient
}

type ExternalVolumeEventOperations interface {
	List(opts *ListOpts) (*ExternalVolumeEventCollection, error)
	Create(opts *ExternalVolumeEvent) (*ExternalVolumeEvent, error)
	Update(existing *ExternalVolumeEvent, updates interface{}) (*ExternalVolumeEvent, error)
	ById(id string) (*ExternalVolumeEvent, error)
	Delete(container *ExternalVolumeEvent) error

	ActionCreate(*ExternalVolumeEvent) (*ExternalEvent, error)

	ActionRemove(*ExternalVolumeEvent) (*ExternalEvent, error)
}

func newExternalVolumeEventClient(rancherClient *RancherClient) *ExternalVolumeEventClient {
	return &ExternalVolumeEventClient{
		rancherClient: rancherClient,
	}
}

func (c *ExternalVolumeEventClient) Create(container *ExternalVolumeEvent) (*ExternalVolumeEvent, error) {
	resp := &ExternalVolumeEvent{}
	err := c.rancherClient.doCreate(EXTERNAL_VOLUME_EVENT_TYPE, container, resp)
	return resp, err
}

func (c *ExternalVolumeEventClient) Update(existing *ExternalVolumeEvent, updates interface{}) (*ExternalVolumeEvent, error) {
	resp := &ExternalVolumeEvent{}
	err := c.rancherClient.doUpdate(EXTERNAL_VOLUME_EVENT_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ExternalVolumeEventClient) List(opts *ListOpts) (*ExternalVolumeEventCollection, error) {
	resp := &ExternalVolumeEventCollection{}
	err := c.rancherClient.doList(EXTERNAL_VOLUME_EVENT_TYPE, opts, resp)
	return resp, err
}

func (c *ExternalVolumeEventClient) ById(id string) (*ExternalVolumeEvent, error) {
	resp := &ExternalVolumeEvent{}
	err := c.rancherClient.doById(EXTERNAL_VOLUME_EVENT_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ExternalVolumeEventClient) Delete(container *ExternalVolumeEvent) error {
	return c.rancherClient.doResourceDelete(EXTERNAL_VOLUME_EVENT_TYPE, &container.Resource)
}

func (c *ExternalVolumeEventClient) ActionCreate(resource *ExternalVolumeEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_VOLUME_EVENT_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *ExternalVolumeEventClient) ActionRemove(resource *ExternalVolumeEvent) (*ExternalEvent, error) {

	resp := &ExternalEvent{}

	err := c.rancherClient.doAction(EXTERNAL_VOLUME_EVENT_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}
