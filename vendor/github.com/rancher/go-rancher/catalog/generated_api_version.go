package catalog

const (
	API_VERSION_TYPE = "apiVersion"
)

type ApiVersion struct {
	Resource

	Actions map[string]interface{} `json:"actions,omitempty" yaml:"actions,omitempty"`

	Links map[string]interface{} `json:"links,omitempty" yaml:"links,omitempty"`

	Type string `json:"type,omitempty" yaml:"type,omitempty"`
}

type ApiVersionCollection struct {
	Collection
	Data []ApiVersion `json:"data,omitempty"`
}

type ApiVersionClient struct {
	rancherClient *RancherClient
}

type ApiVersionOperations interface {
	List(opts *ListOpts) (*ApiVersionCollection, error)
	Create(opts *ApiVersion) (*ApiVersion, error)
	Update(existing *ApiVersion, updates interface{}) (*ApiVersion, error)
	ById(id string) (*ApiVersion, error)
	Delete(container *ApiVersion) error
}

func newApiVersionClient(rancherClient *RancherClient) *ApiVersionClient {
	return &ApiVersionClient{
		rancherClient: rancherClient,
	}
}

func (c *ApiVersionClient) Create(container *ApiVersion) (*ApiVersion, error) {
	resp := &ApiVersion{}
	err := c.rancherClient.doCreate(API_VERSION_TYPE, container, resp)
	return resp, err
}

func (c *ApiVersionClient) Update(existing *ApiVersion, updates interface{}) (*ApiVersion, error) {
	resp := &ApiVersion{}
	err := c.rancherClient.doUpdate(API_VERSION_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ApiVersionClient) List(opts *ListOpts) (*ApiVersionCollection, error) {
	resp := &ApiVersionCollection{}
	err := c.rancherClient.doList(API_VERSION_TYPE, opts, resp)
	return resp, err
}

func (c *ApiVersionClient) ById(id string) (*ApiVersion, error) {
	resp := &ApiVersion{}
	err := c.rancherClient.doById(API_VERSION_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ApiVersionClient) Delete(container *ApiVersion) error {
	return c.rancherClient.doResourceDelete(API_VERSION_TYPE, &container.Resource)
}
