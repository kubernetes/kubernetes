package client

const (
	BINDING_TYPE = "binding"
)

type Binding struct {
	Resource

	Services map[string]interface{} `json:"services,omitempty" yaml:"services,omitempty"`
}

type BindingCollection struct {
	Collection
	Data []Binding `json:"data,omitempty"`
}

type BindingClient struct {
	rancherClient *RancherClient
}

type BindingOperations interface {
	List(opts *ListOpts) (*BindingCollection, error)
	Create(opts *Binding) (*Binding, error)
	Update(existing *Binding, updates interface{}) (*Binding, error)
	ById(id string) (*Binding, error)
	Delete(container *Binding) error
}

func newBindingClient(rancherClient *RancherClient) *BindingClient {
	return &BindingClient{
		rancherClient: rancherClient,
	}
}

func (c *BindingClient) Create(container *Binding) (*Binding, error) {
	resp := &Binding{}
	err := c.rancherClient.doCreate(BINDING_TYPE, container, resp)
	return resp, err
}

func (c *BindingClient) Update(existing *Binding, updates interface{}) (*Binding, error) {
	resp := &Binding{}
	err := c.rancherClient.doUpdate(BINDING_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *BindingClient) List(opts *ListOpts) (*BindingCollection, error) {
	resp := &BindingCollection{}
	err := c.rancherClient.doList(BINDING_TYPE, opts, resp)
	return resp, err
}

func (c *BindingClient) ById(id string) (*Binding, error) {
	resp := &Binding{}
	err := c.rancherClient.doById(BINDING_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *BindingClient) Delete(container *Binding) error {
	return c.rancherClient.doResourceDelete(BINDING_TYPE, &container.Resource)
}
