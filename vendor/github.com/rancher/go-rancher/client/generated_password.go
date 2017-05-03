package client

const (
	PASSWORD_TYPE = "password"
)

type Password struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	PublicValue string `json:"publicValue,omitempty" yaml:"public_value,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	SecretValue string `json:"secretValue,omitempty" yaml:"secret_value,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type PasswordCollection struct {
	Collection
	Data []Password `json:"data,omitempty"`
}

type PasswordClient struct {
	rancherClient *RancherClient
}

type PasswordOperations interface {
	List(opts *ListOpts) (*PasswordCollection, error)
	Create(opts *Password) (*Password, error)
	Update(existing *Password, updates interface{}) (*Password, error)
	ById(id string) (*Password, error)
	Delete(container *Password) error

	ActionActivate(*Password) (*Credential, error)

	ActionChangesecret(*Password, *ChangeSecretInput) (*ChangeSecretInput, error)

	ActionCreate(*Password) (*Credential, error)

	ActionDeactivate(*Password) (*Credential, error)

	ActionPurge(*Password) (*Credential, error)

	ActionRemove(*Password) (*Credential, error)

	ActionUpdate(*Password) (*Credential, error)
}

func newPasswordClient(rancherClient *RancherClient) *PasswordClient {
	return &PasswordClient{
		rancherClient: rancherClient,
	}
}

func (c *PasswordClient) Create(container *Password) (*Password, error) {
	resp := &Password{}
	err := c.rancherClient.doCreate(PASSWORD_TYPE, container, resp)
	return resp, err
}

func (c *PasswordClient) Update(existing *Password, updates interface{}) (*Password, error) {
	resp := &Password{}
	err := c.rancherClient.doUpdate(PASSWORD_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *PasswordClient) List(opts *ListOpts) (*PasswordCollection, error) {
	resp := &PasswordCollection{}
	err := c.rancherClient.doList(PASSWORD_TYPE, opts, resp)
	return resp, err
}

func (c *PasswordClient) ById(id string) (*Password, error) {
	resp := &Password{}
	err := c.rancherClient.doById(PASSWORD_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *PasswordClient) Delete(container *Password) error {
	return c.rancherClient.doResourceDelete(PASSWORD_TYPE, &container.Resource)
}

func (c *PasswordClient) ActionActivate(resource *Password) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(PASSWORD_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PasswordClient) ActionChangesecret(resource *Password, input *ChangeSecretInput) (*ChangeSecretInput, error) {

	resp := &ChangeSecretInput{}

	err := c.rancherClient.doAction(PASSWORD_TYPE, "changesecret", &resource.Resource, input, resp)

	return resp, err
}

func (c *PasswordClient) ActionCreate(resource *Password) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(PASSWORD_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PasswordClient) ActionDeactivate(resource *Password) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(PASSWORD_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PasswordClient) ActionPurge(resource *Password) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(PASSWORD_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PasswordClient) ActionRemove(resource *Password) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(PASSWORD_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *PasswordClient) ActionUpdate(resource *Password) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(PASSWORD_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
