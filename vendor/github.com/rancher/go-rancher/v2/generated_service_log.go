package client

const (
	SERVICE_LOG_TYPE = "serviceLog"
)

type ServiceLog struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	EndTime string `json:"endTime,omitempty" yaml:"end_time,omitempty"`

	EventType string `json:"eventType,omitempty" yaml:"event_type,omitempty"`

	InstanceId string `json:"instanceId,omitempty" yaml:"instance_id,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Level string `json:"level,omitempty" yaml:"level,omitempty"`

	ServiceId string `json:"serviceId,omitempty" yaml:"service_id,omitempty"`

	SubLog bool `json:"subLog,omitempty" yaml:"sub_log,omitempty"`

	TransactionId string `json:"transactionId,omitempty" yaml:"transaction_id,omitempty"`
}

type ServiceLogCollection struct {
	Collection
	Data []ServiceLog `json:"data,omitempty"`
}

type ServiceLogClient struct {
	rancherClient *RancherClient
}

type ServiceLogOperations interface {
	List(opts *ListOpts) (*ServiceLogCollection, error)
	Create(opts *ServiceLog) (*ServiceLog, error)
	Update(existing *ServiceLog, updates interface{}) (*ServiceLog, error)
	ById(id string) (*ServiceLog, error)
	Delete(container *ServiceLog) error
}

func newServiceLogClient(rancherClient *RancherClient) *ServiceLogClient {
	return &ServiceLogClient{
		rancherClient: rancherClient,
	}
}

func (c *ServiceLogClient) Create(container *ServiceLog) (*ServiceLog, error) {
	resp := &ServiceLog{}
	err := c.rancherClient.doCreate(SERVICE_LOG_TYPE, container, resp)
	return resp, err
}

func (c *ServiceLogClient) Update(existing *ServiceLog, updates interface{}) (*ServiceLog, error) {
	resp := &ServiceLog{}
	err := c.rancherClient.doUpdate(SERVICE_LOG_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *ServiceLogClient) List(opts *ListOpts) (*ServiceLogCollection, error) {
	resp := &ServiceLogCollection{}
	err := c.rancherClient.doList(SERVICE_LOG_TYPE, opts, resp)
	return resp, err
}

func (c *ServiceLogClient) ById(id string) (*ServiceLog, error) {
	resp := &ServiceLog{}
	err := c.rancherClient.doById(SERVICE_LOG_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *ServiceLogClient) Delete(container *ServiceLog) error {
	return c.rancherClient.doResourceDelete(SERVICE_LOG_TYPE, &container.Resource)
}
