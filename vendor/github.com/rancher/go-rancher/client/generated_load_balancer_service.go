package client

const (
	LOAD_BALANCER_SERVICE_TYPE = "loadBalancerService"
)

type LoadBalancerService struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	AssignServiceIpAddress bool `json:"assignServiceIpAddress,omitempty" yaml:"assign_service_ip_address,omitempty"`

	CertificateIds []string `json:"certificateIds,omitempty" yaml:"certificate_ids,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	CurrentScale int64 `json:"currentScale,omitempty" yaml:"current_scale,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	DefaultCertificateId string `json:"defaultCertificateId,omitempty" yaml:"default_certificate_id,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	EnvironmentId string `json:"environmentId,omitempty" yaml:"environment_id,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	Fqdn string `json:"fqdn,omitempty" yaml:"fqdn,omitempty"`

	HealthState string `json:"healthState,omitempty" yaml:"health_state,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	LaunchConfig *LaunchConfig `json:"launchConfig,omitempty" yaml:"launch_config,omitempty"`

	LoadBalancerConfig *LoadBalancerConfig `json:"loadBalancerConfig,omitempty" yaml:"load_balancer_config,omitempty"`

	Metadata map[string]interface{} `json:"metadata,omitempty" yaml:"metadata,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	PublicEndpoints []interface{} `json:"publicEndpoints,omitempty" yaml:"public_endpoints,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	RetainIp bool `json:"retainIp,omitempty" yaml:"retain_ip,omitempty"`

	Scale int64 `json:"scale,omitempty" yaml:"scale,omitempty"`

	ScalePolicy *ScalePolicy `json:"scalePolicy,omitempty" yaml:"scale_policy,omitempty"`

	SelectorLink string `json:"selectorLink,omitempty" yaml:"selector_link,omitempty"`

	StartOnCreate bool `json:"startOnCreate,omitempty" yaml:"start_on_create,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Upgrade *ServiceUpgrade `json:"upgrade,omitempty" yaml:"upgrade,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`

	Vip string `json:"vip,omitempty" yaml:"vip,omitempty"`
}

type LoadBalancerServiceCollection struct {
	Collection
	Data []LoadBalancerService `json:"data,omitempty"`
}

type LoadBalancerServiceClient struct {
	rancherClient *RancherClient
}

type LoadBalancerServiceOperations interface {
	List(opts *ListOpts) (*LoadBalancerServiceCollection, error)
	Create(opts *LoadBalancerService) (*LoadBalancerService, error)
	Update(existing *LoadBalancerService, updates interface{}) (*LoadBalancerService, error)
	ById(id string) (*LoadBalancerService, error)
	Delete(container *LoadBalancerService) error

	ActionActivate(*LoadBalancerService) (*Service, error)

	ActionAddservicelink(*LoadBalancerService, *AddRemoveLoadBalancerServiceLinkInput) (*Service, error)

	ActionCancelrollback(*LoadBalancerService) (*Service, error)

	ActionCancelupgrade(*LoadBalancerService) (*Service, error)

	ActionCreate(*LoadBalancerService) (*Service, error)

	ActionDeactivate(*LoadBalancerService) (*Service, error)

	ActionFinishupgrade(*LoadBalancerService) (*Service, error)

	ActionRemove(*LoadBalancerService) (*Service, error)

	ActionRemoveservicelink(*LoadBalancerService, *AddRemoveLoadBalancerServiceLinkInput) (*Service, error)

	ActionRestart(*LoadBalancerService, *ServiceRestart) (*Service, error)

	ActionRollback(*LoadBalancerService) (*Service, error)

	ActionSetservicelinks(*LoadBalancerService, *SetLoadBalancerServiceLinksInput) (*Service, error)

	ActionUpdate(*LoadBalancerService) (*Service, error)

	ActionUpgrade(*LoadBalancerService, *ServiceUpgrade) (*Service, error)
}

func newLoadBalancerServiceClient(rancherClient *RancherClient) *LoadBalancerServiceClient {
	return &LoadBalancerServiceClient{
		rancherClient: rancherClient,
	}
}

func (c *LoadBalancerServiceClient) Create(container *LoadBalancerService) (*LoadBalancerService, error) {
	resp := &LoadBalancerService{}
	err := c.rancherClient.doCreate(LOAD_BALANCER_SERVICE_TYPE, container, resp)
	return resp, err
}

func (c *LoadBalancerServiceClient) Update(existing *LoadBalancerService, updates interface{}) (*LoadBalancerService, error) {
	resp := &LoadBalancerService{}
	err := c.rancherClient.doUpdate(LOAD_BALANCER_SERVICE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *LoadBalancerServiceClient) List(opts *ListOpts) (*LoadBalancerServiceCollection, error) {
	resp := &LoadBalancerServiceCollection{}
	err := c.rancherClient.doList(LOAD_BALANCER_SERVICE_TYPE, opts, resp)
	return resp, err
}

func (c *LoadBalancerServiceClient) ById(id string) (*LoadBalancerService, error) {
	resp := &LoadBalancerService{}
	err := c.rancherClient.doById(LOAD_BALANCER_SERVICE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *LoadBalancerServiceClient) Delete(container *LoadBalancerService) error {
	return c.rancherClient.doResourceDelete(LOAD_BALANCER_SERVICE_TYPE, &container.Resource)
}

func (c *LoadBalancerServiceClient) ActionActivate(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "activate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionAddservicelink(resource *LoadBalancerService, input *AddRemoveLoadBalancerServiceLinkInput) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "addservicelink", &resource.Resource, input, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionCancelrollback(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "cancelrollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionCancelupgrade(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "cancelupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionCreate(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionDeactivate(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "deactivate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionFinishupgrade(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "finishupgrade", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionRemove(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionRemoveservicelink(resource *LoadBalancerService, input *AddRemoveLoadBalancerServiceLinkInput) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "removeservicelink", &resource.Resource, input, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionRestart(resource *LoadBalancerService, input *ServiceRestart) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "restart", &resource.Resource, input, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionRollback(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "rollback", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionSetservicelinks(resource *LoadBalancerService, input *SetLoadBalancerServiceLinksInput) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "setservicelinks", &resource.Resource, input, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionUpdate(resource *LoadBalancerService) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}

func (c *LoadBalancerServiceClient) ActionUpgrade(resource *LoadBalancerService, input *ServiceUpgrade) (*Service, error) {

	resp := &Service{}

	err := c.rancherClient.doAction(LOAD_BALANCER_SERVICE_TYPE, "upgrade", &resource.Resource, input, resp)

	return resp, err
}
