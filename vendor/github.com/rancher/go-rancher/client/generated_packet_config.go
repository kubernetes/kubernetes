package client

const (
	PACKET_CONFIG_TYPE = "packetConfig"
)

type PacketConfig struct {
	Resource

	ApiKey string `json:"apiKey,omitempty" yaml:"api_key,omitempty"`

	BillingCycle string `json:"billingCycle,omitempty" yaml:"billing_cycle,omitempty"`

	FacilityCode string `json:"facilityCode,omitempty" yaml:"facility_code,omitempty"`

	Os string `json:"os,omitempty" yaml:"os,omitempty"`

	Plan string `json:"plan,omitempty" yaml:"plan,omitempty"`

	ProjectId string `json:"projectId,omitempty" yaml:"project_id,omitempty"`
}

type PacketConfigCollection struct {
	Collection
	Data []PacketConfig `json:"data,omitempty"`
}

type PacketConfigClient struct {
	rancherClient *RancherClient
}

type PacketConfigOperations interface {
	List(opts *ListOpts) (*PacketConfigCollection, error)
	Create(opts *PacketConfig) (*PacketConfig, error)
	Update(existing *PacketConfig, updates interface{}) (*PacketConfig, error)
	ById(id string) (*PacketConfig, error)
	Delete(container *PacketConfig) error
}

func newPacketConfigClient(rancherClient *RancherClient) *PacketConfigClient {
	return &PacketConfigClient{
		rancherClient: rancherClient,
	}
}

func (c *PacketConfigClient) Create(container *PacketConfig) (*PacketConfig, error) {
	resp := &PacketConfig{}
	err := c.rancherClient.doCreate(PACKET_CONFIG_TYPE, container, resp)
	return resp, err
}

func (c *PacketConfigClient) Update(existing *PacketConfig, updates interface{}) (*PacketConfig, error) {
	resp := &PacketConfig{}
	err := c.rancherClient.doUpdate(PACKET_CONFIG_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *PacketConfigClient) List(opts *ListOpts) (*PacketConfigCollection, error) {
	resp := &PacketConfigCollection{}
	err := c.rancherClient.doList(PACKET_CONFIG_TYPE, opts, resp)
	return resp, err
}

func (c *PacketConfigClient) ById(id string) (*PacketConfig, error) {
	resp := &PacketConfig{}
	err := c.rancherClient.doById(PACKET_CONFIG_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *PacketConfigClient) Delete(container *PacketConfig) error {
	return c.rancherClient.doResourceDelete(PACKET_CONFIG_TYPE, &container.Resource)
}
