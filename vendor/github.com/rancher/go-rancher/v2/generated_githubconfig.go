package client

const (
	GITHUBCONFIG_TYPE = "githubconfig"
)

type Githubconfig struct {
	Resource

	AccessMode string `json:"accessMode,omitempty" yaml:"access_mode,omitempty"`

	AllowedIdentities []interface{} `json:"allowedIdentities,omitempty" yaml:"allowed_identities,omitempty"`

	ClientId string `json:"clientId,omitempty" yaml:"client_id,omitempty"`

	ClientSecret string `json:"clientSecret,omitempty" yaml:"client_secret,omitempty"`

	Enabled bool `json:"enabled,omitempty" yaml:"enabled,omitempty"`

	Hostname string `json:"hostname,omitempty" yaml:"hostname,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Scheme string `json:"scheme,omitempty" yaml:"scheme,omitempty"`
}

type GithubconfigCollection struct {
	Collection
	Data []Githubconfig `json:"data,omitempty"`
}

type GithubconfigClient struct {
	rancherClient *RancherClient
}

type GithubconfigOperations interface {
	List(opts *ListOpts) (*GithubconfigCollection, error)
	Create(opts *Githubconfig) (*Githubconfig, error)
	Update(existing *Githubconfig, updates interface{}) (*Githubconfig, error)
	ById(id string) (*Githubconfig, error)
	Delete(container *Githubconfig) error
}

func newGithubconfigClient(rancherClient *RancherClient) *GithubconfigClient {
	return &GithubconfigClient{
		rancherClient: rancherClient,
	}
}

func (c *GithubconfigClient) Create(container *Githubconfig) (*Githubconfig, error) {
	resp := &Githubconfig{}
	err := c.rancherClient.doCreate(GITHUBCONFIG_TYPE, container, resp)
	return resp, err
}

func (c *GithubconfigClient) Update(existing *Githubconfig, updates interface{}) (*Githubconfig, error) {
	resp := &Githubconfig{}
	err := c.rancherClient.doUpdate(GITHUBCONFIG_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *GithubconfigClient) List(opts *ListOpts) (*GithubconfigCollection, error) {
	resp := &GithubconfigCollection{}
	err := c.rancherClient.doList(GITHUBCONFIG_TYPE, opts, resp)
	return resp, err
}

func (c *GithubconfigClient) ById(id string) (*Githubconfig, error) {
	resp := &Githubconfig{}
	err := c.rancherClient.doById(GITHUBCONFIG_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *GithubconfigClient) Delete(container *Githubconfig) error {
	return c.rancherClient.doResourceDelete(GITHUBCONFIG_TYPE, &container.Resource)
}
