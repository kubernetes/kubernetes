package client

const (
	DATABASECHANGELOG_TYPE = "databasechangelog"
)

type Databasechangelog struct {
	Resource

	Author string `json:"author,omitempty" yaml:"author,omitempty"`

	Comments string `json:"comments,omitempty" yaml:"comments,omitempty"`

	Dateexecuted string `json:"dateexecuted,omitempty" yaml:"dateexecuted,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Exectype string `json:"exectype,omitempty" yaml:"exectype,omitempty"`

	Filename string `json:"filename,omitempty" yaml:"filename,omitempty"`

	Liquibase string `json:"liquibase,omitempty" yaml:"liquibase,omitempty"`

	Md5sum string `json:"md5sum,omitempty" yaml:"md5sum,omitempty"`

	Orderexecuted int64 `json:"orderexecuted,omitempty" yaml:"orderexecuted,omitempty"`

	Tag string `json:"tag,omitempty" yaml:"tag,omitempty"`
}

type DatabasechangelogCollection struct {
	Collection
	Data []Databasechangelog `json:"data,omitempty"`
}

type DatabasechangelogClient struct {
	rancherClient *RancherClient
}

type DatabasechangelogOperations interface {
	List(opts *ListOpts) (*DatabasechangelogCollection, error)
	Create(opts *Databasechangelog) (*Databasechangelog, error)
	Update(existing *Databasechangelog, updates interface{}) (*Databasechangelog, error)
	ById(id string) (*Databasechangelog, error)
	Delete(container *Databasechangelog) error
}

func newDatabasechangelogClient(rancherClient *RancherClient) *DatabasechangelogClient {
	return &DatabasechangelogClient{
		rancherClient: rancherClient,
	}
}

func (c *DatabasechangelogClient) Create(container *Databasechangelog) (*Databasechangelog, error) {
	resp := &Databasechangelog{}
	err := c.rancherClient.doCreate(DATABASECHANGELOG_TYPE, container, resp)
	return resp, err
}

func (c *DatabasechangelogClient) Update(existing *Databasechangelog, updates interface{}) (*Databasechangelog, error) {
	resp := &Databasechangelog{}
	err := c.rancherClient.doUpdate(DATABASECHANGELOG_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *DatabasechangelogClient) List(opts *ListOpts) (*DatabasechangelogCollection, error) {
	resp := &DatabasechangelogCollection{}
	err := c.rancherClient.doList(DATABASECHANGELOG_TYPE, opts, resp)
	return resp, err
}

func (c *DatabasechangelogClient) ById(id string) (*Databasechangelog, error) {
	resp := &Databasechangelog{}
	err := c.rancherClient.doById(DATABASECHANGELOG_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *DatabasechangelogClient) Delete(container *Databasechangelog) error {
	return c.rancherClient.doResourceDelete(DATABASECHANGELOG_TYPE, &container.Resource)
}
