package client

const (
	REGISTRATION_TOKEN_TYPE = "registrationToken"
)

type RegistrationToken struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Command string `json:"command,omitempty" yaml:"command,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Image string `json:"image,omitempty" yaml:"image,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	RegistrationUrl string `json:"registrationUrl,omitempty" yaml:"registration_url,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Token string `json:"token,omitempty" yaml:"token,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type RegistrationTokenCollection struct {
	Collection
	Data []RegistrationToken `json:"data,omitempty"`
}

type RegistrationTokenClient struct {
	rancherClient *RancherClient
}

type RegistrationTokenOperations interface {
	List(opts *ListOpts) (*RegistrationTokenCollection, error)
	Create(opts *RegistrationToken) (*RegistrationToken, error)
	Update(existing *RegistrationToken, updates interface{}) (*RegistrationToken, error)
	ById(id string) (*RegistrationToken, error)
	Delete(container *RegistrationToken) error

	ActionActivate(*RegistrationToken) (*Credential, error)

	ActionCreate(*RegistrationToken) (*Credential, error)

	ActionDeactivate(*RegistrationToken) (*Credential, error)

	ActionPurge(*RegistrationToken) (*Credential, error)

	ActionRemove(*RegistrationToken) (*Credential, error)

	ActionUpdate(*RegistrationToken) (*Credential, error)
}

func newRegistrationTokenClient(rancherClient *RancherClient) *RegistrationTokenClient {
	return &RegistrationTokenClient{
		rancherClient: rancherClient,
	}
}

func (c *RegistrationTokenClient) Create(container *RegistrationToken) (*RegistrationToken, error) {
	resp := &RegistrationToken{}
	err := c.rancherClient.doCreate(REGISTRATION_TOKEN_TYPE, container, resp)
	return resp, err
}

func (c *RegistrationTokenClient) Update(existing *RegistrationToken, updates interface{}) (*RegistrationToken, error) {
	resp := &RegistrationToken{}
	err := c.rancherClient.doUpdate(REGISTRATION_TOKEN_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *RegistrationTokenClient) List(opts *ListOpts) (*RegistrationTokenCollection, error) {
	resp := &RegistrationTokenCollection{}
	err := c.rancherClient.doList(REGISTRATION_TOKEN_TYPE, opts, resp)
	return resp, err
}

func (c *RegistrationTokenClient) ById(id string) (*RegistrationToken, error) {
	resp := &RegistrationToken{}
	err := c.rancherClient.doById(REGISTRATION_TOKEN_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *RegistrationTokenClient) Delete(container *RegistrationToken) error {
	return c.rancherClient.doResourceDelete(REGISTRATION_TOKEN_TYPE, &container.Resource)
}

func (c *RegistrationTokenClient) ActionActivate(resource *RegistrationToken) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRATION_TOKEN_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistrationTokenClient) ActionCreate(resource *RegistrationToken) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRATION_TOKEN_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistrationTokenClient) ActionDeactivate(resource *RegistrationToken) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRATION_TOKEN_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistrationTokenClient) ActionPurge(resource *RegistrationToken) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRATION_TOKEN_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistrationTokenClient) ActionRemove(resource *RegistrationToken) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRATION_TOKEN_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistrationTokenClient) ActionUpdate(resource *RegistrationToken) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRATION_TOKEN_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
