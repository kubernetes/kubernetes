package client

const (
	VIRTUAL_MACHINE_DISK_TYPE = "virtualMachineDisk"
)

type VirtualMachineDisk struct {
	Resource

	Driver string `json:"driver,omitempty" yaml:"driver,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	Opts map[string]interface{} `json:"opts,omitempty" yaml:"opts,omitempty"`

	ReadIops int64 `json:"readIops,omitempty" yaml:"read_iops,omitempty"`

	Root bool `json:"root,omitempty" yaml:"root,omitempty"`

	Size string `json:"size,omitempty" yaml:"size,omitempty"`

	WriteIops int64 `json:"writeIops,omitempty" yaml:"write_iops,omitempty"`
}

type VirtualMachineDiskCollection struct {
	Collection
	Data []VirtualMachineDisk `json:"data,omitempty"`
}

type VirtualMachineDiskClient struct {
	rancherClient *RancherClient
}

type VirtualMachineDiskOperations interface {
	List(opts *ListOpts) (*VirtualMachineDiskCollection, error)
	Create(opts *VirtualMachineDisk) (*VirtualMachineDisk, error)
	Update(existing *VirtualMachineDisk, updates interface{}) (*VirtualMachineDisk, error)
	ById(id string) (*VirtualMachineDisk, error)
	Delete(container *VirtualMachineDisk) error
}

func newVirtualMachineDiskClient(rancherClient *RancherClient) *VirtualMachineDiskClient {
	return &VirtualMachineDiskClient{
		rancherClient: rancherClient,
	}
}

func (c *VirtualMachineDiskClient) Create(container *VirtualMachineDisk) (*VirtualMachineDisk, error) {
	resp := &VirtualMachineDisk{}
	err := c.rancherClient.doCreate(VIRTUAL_MACHINE_DISK_TYPE, container, resp)
	return resp, err
}

func (c *VirtualMachineDiskClient) Update(existing *VirtualMachineDisk, updates interface{}) (*VirtualMachineDisk, error) {
	resp := &VirtualMachineDisk{}
	err := c.rancherClient.doUpdate(VIRTUAL_MACHINE_DISK_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *VirtualMachineDiskClient) List(opts *ListOpts) (*VirtualMachineDiskCollection, error) {
	resp := &VirtualMachineDiskCollection{}
	err := c.rancherClient.doList(VIRTUAL_MACHINE_DISK_TYPE, opts, resp)
	return resp, err
}

func (c *VirtualMachineDiskClient) ById(id string) (*VirtualMachineDisk, error) {
	resp := &VirtualMachineDisk{}
	err := c.rancherClient.doById(VIRTUAL_MACHINE_DISK_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *VirtualMachineDiskClient) Delete(container *VirtualMachineDisk) error {
	return c.rancherClient.doResourceDelete(VIRTUAL_MACHINE_DISK_TYPE, &container.Resource)
}
