package client

const (
	ACTIVE_SETTING_TYPE = "activeSetting"
)

type ActiveSetting struct {
	Resource

	ActiveValue interface{} `json:"activeValue,omitempty" yaml:"active_value,omitempty"`

	InDb bool `json:"inDb,omitempty" yaml:"in_db,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Source string `json:"source,omitempty" yaml:"source,omitempty"`

	Value string `json:"value,omitempty" yaml:"value,omitempty"`
}

type ActiveSettingCollection struct {
	Collection
	Data []ActiveSetting `json:"data,omitempty"`
}

type ActiveSettingClient struct {
	rancherClient *RancherClient
}

type ActiveSettingOperations interface {
	List(opts *ListOpts) (*ActiveSettingCollection, error)
	Create(opts *ActiveSetting) (*ActiveSetting, error)
	Update(existing *ActiveSetting, updates interface{}) (*ActiveSetting, error)
	ById(id string) (*ActiveSetting, error)
	Delete(container *ActiveSetting) error
}

func newActiveSettingClient(rancherClient *RancherClient) *ActiveSettingClient {
	return &ActiveSettingClient{
		rancherClient: rancherClient,
	}
}

func (c *ActiveSettingClient) Create(container *ActiveSetting) (*ActiveSetting, error) {
	resp := &ActiveSetting{}
	err := c.rancherClient.doCreate(ACTIVE_SETTING_TYPE, container, resp)
	return resp, err
}

func (c *ActiveSettingClient) Update(existing *ActiveSetting, updates interface{}) (*ActiveSetting, error) {
	resp := &ActiveSetting{}
	err := c.rancherClient.doUpdate(ACTIVE_SETTING_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ActiveSettingClient) List(opts *ListOpts) (*ActiveSettingCollection, error) {
	resp := &ActiveSettingCollection{}
	err := c.rancherClient.doList(ACTIVE_SETTING_TYPE, opts, resp)
	return resp, err
}

func (c *ActiveSettingClient) ById(id string) (*ActiveSetting, error) {
	resp := &ActiveSetting{}
	err := c.rancherClient.doById(ACTIVE_SETTING_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ActiveSettingClient) Delete(container *ActiveSetting) error {
	return c.rancherClient.doResourceDelete(ACTIVE_SETTING_TYPE, &container.Resource)
}
