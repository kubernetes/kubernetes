package client

const (
	SETTING_TYPE = "setting"
)

type Setting struct {
	Resource

	ActiveValue string `json:"activeValue,omitempty" yaml:"active_value,omitempty"`

	InDb bool `json:"inDb,omitempty" yaml:"in_db,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Source string `json:"source,omitempty" yaml:"source,omitempty"`

	Value string `json:"value,omitempty" yaml:"value,omitempty"`
}

type SettingCollection struct {
	Collection
	Data []Setting `json:"data,omitempty"`
}

type SettingClient struct {
	rancherClient *RancherClient
}

type SettingOperations interface {
	List(opts *ListOpts) (*SettingCollection, error)
	Create(opts *Setting) (*Setting, error)
	Update(existing *Setting, updates interface{}) (*Setting, error)
	ById(id string) (*Setting, error)
	Delete(container *Setting) error
}

func newSettingClient(rancherClient *RancherClient) *SettingClient {
	return &SettingClient{
		rancherClient: rancherClient,
	}
}

func (c *SettingClient) Create(container *Setting) (*Setting, error) {
	resp := &Setting{}
	err := c.rancherClient.doCreate(SETTING_TYPE, container, resp)
	return resp, err
}

func (c *SettingClient) Update(existing *Setting, updates interface{}) (*Setting, error) {
	resp := &Setting{}
	err := c.rancherClient.doUpdate(SETTING_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *SettingClient) List(opts *ListOpts) (*SettingCollection, error) {
	resp := &SettingCollection{}
	err := c.rancherClient.doList(SETTING_TYPE, opts, resp)
	return resp, err
}

func (c *SettingClient) ById(id string) (*Setting, error) {
	resp := &Setting{}
	err := c.rancherClient.doById(SETTING_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *SettingClient) Delete(container *Setting) error {
	return c.rancherClient.doResourceDelete(SETTING_TYPE, &container.Resource)
}
