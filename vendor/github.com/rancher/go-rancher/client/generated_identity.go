package client

const (
	IDENTITY_TYPE = "identity"
)

type Identity struct {
	Resource

	All string `json:"all,omitempty" yaml:"all,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	ExternalIdType string `json:"externalIdType,omitempty" yaml:"external_id_type,omitempty"`

	Login string `json:"login,omitempty" yaml:"login,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	ProfilePicture string `json:"profilePicture,omitempty" yaml:"profile_picture,omitempty"`

	ProfileUrl string `json:"profileUrl,omitempty" yaml:"profile_url,omitempty"`

	ProjectId string `json:"projectId,omitempty" yaml:"project_id,omitempty"`

	Role string `json:"role,omitempty" yaml:"role,omitempty"`
}

type IdentityCollection struct {
	Collection
	Data []Identity `json:"data,omitempty"`
}

type IdentityClient struct {
	rancherClient *RancherClient
}

type IdentityOperations interface {
	List(opts *ListOpts) (*IdentityCollection, error)
	Create(opts *Identity) (*Identity, error)
	Update(existing *Identity, updates interface{}) (*Identity, error)
	ById(id string) (*Identity, error)
	Delete(container *Identity) error
}

func newIdentityClient(rancherClient *RancherClient) *IdentityClient {
	return &IdentityClient{
		rancherClient: rancherClient,
	}
}

func (c *IdentityClient) Create(container *Identity) (*Identity, error) {
	resp := &Identity{}
	err := c.rancherClient.doCreate(IDENTITY_TYPE, container, resp)
	return resp, err
}

func (c *IdentityClient) Update(existing *Identity, updates interface{}) (*Identity, error) {
	resp := &Identity{}
	err := c.rancherClient.doUpdate(IDENTITY_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *IdentityClient) List(opts *ListOpts) (*IdentityCollection, error) {
	resp := &IdentityCollection{}
	err := c.rancherClient.doList(IDENTITY_TYPE, opts, resp)
	return resp, err
}

func (c *IdentityClient) ById(id string) (*Identity, error) {
	resp := &Identity{}
	err := c.rancherClient.doById(IDENTITY_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *IdentityClient) Delete(container *Identity) error {
	return c.rancherClient.doResourceDelete(IDENTITY_TYPE, &container.Resource)
}
