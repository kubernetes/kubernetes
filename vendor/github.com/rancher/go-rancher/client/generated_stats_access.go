package client

const (
	STATS_ACCESS_TYPE = "statsAccess"
)

type StatsAccess struct {
	Resource

	Token string `json:"token,omitempty" yaml:"token,omitempty"`

	Url string `json:"url,omitempty" yaml:"url,omitempty"`
}

type StatsAccessCollection struct {
	Collection
	Data []StatsAccess `json:"data,omitempty"`
}

type StatsAccessClient struct {
	rancherClient *RancherClient
}

type StatsAccessOperations interface {
	List(opts *ListOpts) (*StatsAccessCollection, error)
	Create(opts *StatsAccess) (*StatsAccess, error)
	Update(existing *StatsAccess, updates interface{}) (*StatsAccess, error)
	ById(id string) (*StatsAccess, error)
	Delete(container *StatsAccess) error
}

func newStatsAccessClient(rancherClient *RancherClient) *StatsAccessClient {
	return &StatsAccessClient{
		rancherClient: rancherClient,
	}
}

func (c *StatsAccessClient) Create(container *StatsAccess) (*StatsAccess, error) {
	resp := &StatsAccess{}
	err := c.rancherClient.doCreate(STATS_ACCESS_TYPE, container, resp)
	return resp, err
}

func (c *StatsAccessClient) Update(existing *StatsAccess, updates interface{}) (*StatsAccess, error) {
	resp := &StatsAccess{}
	err := c.rancherClient.doUpdate(STATS_ACCESS_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *StatsAccessClient) List(opts *ListOpts) (*StatsAccessCollection, error) {
	resp := &StatsAccessCollection{}
	err := c.rancherClient.doList(STATS_ACCESS_TYPE, opts, resp)
	return resp, err
}

func (c *StatsAccessClient) ById(id string) (*StatsAccess, error) {
	resp := &StatsAccess{}
	err := c.rancherClient.doById(STATS_ACCESS_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *StatsAccessClient) Delete(container *StatsAccess) error {
	return c.rancherClient.doResourceDelete(STATS_ACCESS_TYPE, &container.Resource)
}
