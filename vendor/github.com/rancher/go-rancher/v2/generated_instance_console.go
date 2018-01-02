package client

const (
	INSTANCE_CONSOLE_TYPE = "instanceConsole"
)

type InstanceConsole struct {
	Resource

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Password string `json:"password,omitempty" yaml:"password,omitempty"`

	Url string `json:"url,omitempty" yaml:"url,omitempty"`
}

type InstanceConsoleCollection struct {
	Collection
	Data []InstanceConsole `json:"data,omitempty"`
}

type InstanceConsoleClient struct {
	rancherClient *RancherClient
}

type InstanceConsoleOperations interface {
	List(opts *ListOpts) (*InstanceConsoleCollection, error)
	Create(opts *InstanceConsole) (*InstanceConsole, error)
	Update(existing *InstanceConsole, updates interface{}) (*InstanceConsole, error)
	ById(id string) (*InstanceConsole, error)
	Delete(container *InstanceConsole) error
}

func newInstanceConsoleClient(rancherClient *RancherClient) *InstanceConsoleClient {
	return &InstanceConsoleClient{
		rancherClient: rancherClient,
	}
}

func (c *InstanceConsoleClient) Create(container *InstanceConsole) (*InstanceConsole, error) {
	resp := &InstanceConsole{}
	err := c.rancherClient.doCreate(INSTANCE_CONSOLE_TYPE, container, resp)
	return resp, err
}

func (c *InstanceConsoleClient) Update(existing *InstanceConsole, updates interface{}) (*InstanceConsole, error) {
	resp := &InstanceConsole{}
	err := c.rancherClient.doUpdate(INSTANCE_CONSOLE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *InstanceConsoleClient) List(opts *ListOpts) (*InstanceConsoleCollection, error) {
	resp := &InstanceConsoleCollection{}
	err := c.rancherClient.doList(INSTANCE_CONSOLE_TYPE, opts, resp)
	return resp, err
}

func (c *InstanceConsoleClient) ById(id string) (*InstanceConsole, error) {
	resp := &InstanceConsole{}
	err := c.rancherClient.doById(INSTANCE_CONSOLE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *InstanceConsoleClient) Delete(container *InstanceConsole) error {
	return c.rancherClient.doResourceDelete(INSTANCE_CONSOLE_TYPE, &container.Resource)
}
