package client

const (
	CREDENTIAL_TYPE = "credential"
)

type Credential struct {
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

type CredentialCollection struct {
	Collection
	Data []Credential `json:"data,omitempty"`
}

type CredentialClient struct {
	rancherClient *RancherClient
}

type CredentialOperations interface {
	List(opts *ListOpts) (*CredentialCollection, error)
	Create(opts *Credential) (*Credential, error)
	Update(existing *Credential, updates interface{}) (*Credential, error)
	ById(id string) (*Credential, error)
	Delete(container *Credential) error

	ActionActivate(*Credential) (*Credential, error)

	ActionCreate(*Credential) (*Credential, error)

	ActionDeactivate(*Credential) (*Credential, error)

	ActionPurge(*Credential) (*Credential, error)

	ActionRemove(*Credential) (*Credential, error)

	ActionUpdate(*Credential) (*Credential, error)
}

func newCredentialClient(rancherClient *RancherClient) *CredentialClient {
	return &CredentialClient{
		rancherClient: rancherClient,
	}
}

func (c *CredentialClient) Create(container *Credential) (*Credential, error) {
	resp := &Credential{}
	err := c.rancherClient.doCreate(CREDENTIAL_TYPE, container, resp)
	return resp, err
}

func (c *CredentialClient) Update(existing *Credential, updates interface{}) (*Credential, error) {
	resp := &Credential{}
	err := c.rancherClient.doUpdate(CREDENTIAL_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *CredentialClient) List(opts *ListOpts) (*CredentialCollection, error) {
	resp := &CredentialCollection{}
	err := c.rancherClient.doList(CREDENTIAL_TYPE, opts, resp)
	return resp, err
}

func (c *CredentialClient) ById(id string) (*Credential, error) {
	resp := &Credential{}
	err := c.rancherClient.doById(CREDENTIAL_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *CredentialClient) Delete(container *Credential) error {
	return c.rancherClient.doResourceDelete(CREDENTIAL_TYPE, &container.Resource)
}

func (c *CredentialClient) ActionActivate(resource *Credential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(CREDENTIAL_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *CredentialClient) ActionCreate(resource *Credential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(CREDENTIAL_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *CredentialClient) ActionDeactivate(resource *Credential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(CREDENTIAL_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *CredentialClient) ActionPurge(resource *Credential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(CREDENTIAL_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *CredentialClient) ActionRemove(resource *Credential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(CREDENTIAL_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *CredentialClient) ActionUpdate(resource *Credential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(CREDENTIAL_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
