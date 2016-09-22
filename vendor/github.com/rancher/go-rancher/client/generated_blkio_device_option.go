package client

const (
	BLKIO_DEVICE_OPTION_TYPE = "blkioDeviceOption"
)

type BlkioDeviceOption struct {
	Resource

	ReadBps int64 `json:"readBps,omitempty" yaml:"read_bps,omitempty"`

	ReadIops int64 `json:"readIops,omitempty" yaml:"read_iops,omitempty"`

	Weight int64 `json:"weight,omitempty" yaml:"weight,omitempty"`

	WriteBps int64 `json:"writeBps,omitempty" yaml:"write_bps,omitempty"`

	WriteIops int64 `json:"writeIops,omitempty" yaml:"write_iops,omitempty"`
}

type BlkioDeviceOptionCollection struct {
	Collection
	Data []BlkioDeviceOption `json:"data,omitempty"`
}

type BlkioDeviceOptionClient struct {
	rancherClient *RancherClient
}

type BlkioDeviceOptionOperations interface {
	List(opts *ListOpts) (*BlkioDeviceOptionCollection, error)
	Create(opts *BlkioDeviceOption) (*BlkioDeviceOption, error)
	Update(existing *BlkioDeviceOption, updates interface{}) (*BlkioDeviceOption, error)
	ById(id string) (*BlkioDeviceOption, error)
	Delete(container *BlkioDeviceOption) error
}

func newBlkioDeviceOptionClient(rancherClient *RancherClient) *BlkioDeviceOptionClient {
	return &BlkioDeviceOptionClient{
		rancherClient: rancherClient,
	}
}

func (c *BlkioDeviceOptionClient) Create(container *BlkioDeviceOption) (*BlkioDeviceOption, error) {
	resp := &BlkioDeviceOption{}
	err := c.rancherClient.doCreate(BLKIO_DEVICE_OPTION_TYPE, container, resp)
	return resp, err
}

func (c *BlkioDeviceOptionClient) Update(existing *BlkioDeviceOption, updates interface{}) (*BlkioDeviceOption, error) {
	resp := &BlkioDeviceOption{}
	err := c.rancherClient.doUpdate(BLKIO_DEVICE_OPTION_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *BlkioDeviceOptionClient) List(opts *ListOpts) (*BlkioDeviceOptionCollection, error) {
	resp := &BlkioDeviceOptionCollection{}
	err := c.rancherClient.doList(BLKIO_DEVICE_OPTION_TYPE, opts, resp)
	return resp, err
}

func (c *BlkioDeviceOptionClient) ById(id string) (*BlkioDeviceOption, error) {
	resp := &BlkioDeviceOption{}
	err := c.rancherClient.doById(BLKIO_DEVICE_OPTION_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *BlkioDeviceOptionClient) Delete(container *BlkioDeviceOption) error {
	return c.rancherClient.doResourceDelete(BLKIO_DEVICE_OPTION_TYPE, &container.Resource)
}
