package catalog

const (
	ERROR_TYPE = "error"
)

type Error struct {
	Resource

	Actions map[string]interface{} `json:"actions,omitempty" yaml:"actions,omitempty"`

	Links map[string]interface{} `json:"links,omitempty" yaml:"links,omitempty"`

	Message string `json:"message,omitempty" yaml:"message,omitempty"`

	Status string `json:"status,omitempty" yaml:"status,omitempty"`

	Type string `json:"type,omitempty" yaml:"type,omitempty"`
}

type ErrorCollection struct {
	Collection
	Data []Error `json:"data,omitempty"`
}

type ErrorClient struct {
	rancherClient *RancherClient
}

type ErrorOperations interface {
	List(opts *ListOpts) (*ErrorCollection, error)
	Create(opts *Error) (*Error, error)
	Update(existing *Error, updates interface{}) (*Error, error)
	ById(id string) (*Error, error)
	Delete(container *Error) error
}

func newErrorClient(rancherClient *RancherClient) *ErrorClient {
	return &ErrorClient{
		rancherClient: rancherClient,
	}
}

func (c *ErrorClient) Create(container *Error) (*Error, error) {
	resp := &Error{}
	err := c.rancherClient.doCreate(ERROR_TYPE, container, resp)
	return resp, err
}

func (c *ErrorClient) Update(existing *Error, updates interface{}) (*Error, error) {
	resp := &Error{}
	err := c.rancherClient.doUpdate(ERROR_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ErrorClient) List(opts *ListOpts) (*ErrorCollection, error) {
	resp := &ErrorCollection{}
	err := c.rancherClient.doList(ERROR_TYPE, opts, resp)
	return resp, err
}

func (c *ErrorClient) ById(id string) (*Error, error) {
	resp := &Error{}
	err := c.rancherClient.doById(ERROR_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ErrorClient) Delete(container *Error) error {
	return c.rancherClient.doResourceDelete(ERROR_TYPE, &container.Resource)
}
