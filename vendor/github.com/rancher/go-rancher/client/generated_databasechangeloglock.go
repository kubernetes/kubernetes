package client

const (
	DATABASECHANGELOGLOCK_TYPE = "databasechangeloglock"
)

type Databasechangeloglock struct {
	Resource

	Locked bool `json:"locked,omitempty" yaml:"locked,omitempty"`

	Lockedby string `json:"lockedby,omitempty" yaml:"lockedby,omitempty"`

	Lockgranted string `json:"lockgranted,omitempty" yaml:"lockgranted,omitempty"`
}

type DatabasechangeloglockCollection struct {
	Collection
	Data []Databasechangeloglock `json:"data,omitempty"`
}

type DatabasechangeloglockClient struct {
	rancherClient *RancherClient
}

type DatabasechangeloglockOperations interface {
	List(opts *ListOpts) (*DatabasechangeloglockCollection, error)
	Create(opts *Databasechangeloglock) (*Databasechangeloglock, error)
	Update(existing *Databasechangeloglock, updates interface{}) (*Databasechangeloglock, error)
	ById(id string) (*Databasechangeloglock, error)
	Delete(container *Databasechangeloglock) error
}

func newDatabasechangeloglockClient(rancherClient *RancherClient) *DatabasechangeloglockClient {
	return &DatabasechangeloglockClient{
		rancherClient: rancherClient,
	}
}

func (c *DatabasechangeloglockClient) Create(container *Databasechangeloglock) (*Databasechangeloglock, error) {
	resp := &Databasechangeloglock{}
	err := c.rancherClient.doCreate(DATABASECHANGELOGLOCK_TYPE, container, resp)
	return resp, err
}

func (c *DatabasechangeloglockClient) Update(existing *Databasechangeloglock, updates interface{}) (*Databasechangeloglock, error) {
	resp := &Databasechangeloglock{}
	err := c.rancherClient.doUpdate(DATABASECHANGELOGLOCK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *DatabasechangeloglockClient) List(opts *ListOpts) (*DatabasechangeloglockCollection, error) {
	resp := &DatabasechangeloglockCollection{}
	err := c.rancherClient.doList(DATABASECHANGELOGLOCK_TYPE, opts, resp)
	return resp, err
}

func (c *DatabasechangeloglockClient) ById(id string) (*Databasechangeloglock, error) {
	resp := &Databasechangeloglock{}
	err := c.rancherClient.doById(DATABASECHANGELOGLOCK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *DatabasechangeloglockClient) Delete(container *Databasechangeloglock) error {
	return c.rancherClient.doResourceDelete(DATABASECHANGELOGLOCK_TYPE, &container.Resource)
}
