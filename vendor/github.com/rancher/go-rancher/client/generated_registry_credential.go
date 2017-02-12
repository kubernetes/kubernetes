package client

const (
	REGISTRY_CREDENTIAL_TYPE = "registryCredential"
)

type RegistryCredential struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Email string `json:"email,omitempty" yaml:"email,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	PublicValue string `json:"publicValue,omitempty" yaml:"public_value,omitempty"`

	RegistryId string `json:"registryId,omitempty" yaml:"registry_id,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	SecretValue string `json:"secretValue,omitempty" yaml:"secret_value,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`
}

type RegistryCredentialCollection struct {
	Collection
	Data []RegistryCredential `json:"data,omitempty"`
}

type RegistryCredentialClient struct {
	rancherClient *RancherClient
}

type RegistryCredentialOperations interface {
	List(opts *ListOpts) (*RegistryCredentialCollection, error)
	Create(opts *RegistryCredential) (*RegistryCredential, error)
	Update(existing *RegistryCredential, updates interface{}) (*RegistryCredential, error)
	ById(id string) (*RegistryCredential, error)
	Delete(container *RegistryCredential) error

	ActionActivate(*RegistryCredential) (*Credential, error)

	ActionCreate(*RegistryCredential) (*Credential, error)

	ActionDeactivate(*RegistryCredential) (*Credential, error)

	ActionPurge(*RegistryCredential) (*Credential, error)

	ActionRemove(*RegistryCredential) (*Credential, error)

	ActionUpdate(*RegistryCredential) (*Credential, error)
}

func newRegistryCredentialClient(rancherClient *RancherClient) *RegistryCredentialClient {
	return &RegistryCredentialClient{
		rancherClient: rancherClient,
	}
}

func (c *RegistryCredentialClient) Create(container *RegistryCredential) (*RegistryCredential, error) {
	resp := &RegistryCredential{}
	err := c.rancherClient.doCreate(REGISTRY_CREDENTIAL_TYPE, container, resp)
	return resp, err
}

func (c *RegistryCredentialClient) Update(existing *RegistryCredential, updates interface{}) (*RegistryCredential, error) {
	resp := &RegistryCredential{}
	err := c.rancherClient.doUpdate(REGISTRY_CREDENTIAL_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *RegistryCredentialClient) List(opts *ListOpts) (*RegistryCredentialCollection, error) {
	resp := &RegistryCredentialCollection{}
	err := c.rancherClient.doList(REGISTRY_CREDENTIAL_TYPE, opts, resp)
	return resp, err
}

func (c *RegistryCredentialClient) ById(id string) (*RegistryCredential, error) {
	resp := &RegistryCredential{}
	err := c.rancherClient.doById(REGISTRY_CREDENTIAL_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *RegistryCredentialClient) Delete(container *RegistryCredential) error {
	return c.rancherClient.doResourceDelete(REGISTRY_CREDENTIAL_TYPE, &container.Resource)
}

func (c *RegistryCredentialClient) ActionActivate(resource *RegistryCredential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRY_CREDENTIAL_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistryCredentialClient) ActionCreate(resource *RegistryCredential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRY_CREDENTIAL_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistryCredentialClient) ActionDeactivate(resource *RegistryCredential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRY_CREDENTIAL_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistryCredentialClient) ActionPurge(resource *RegistryCredential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRY_CREDENTIAL_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistryCredentialClient) ActionRemove(resource *RegistryCredential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRY_CREDENTIAL_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *RegistryCredentialClient) ActionUpdate(resource *RegistryCredential) (*Credential, error) {

	resp := &Credential{}

	err := c.rancherClient.doAction(REGISTRY_CREDENTIAL_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}
