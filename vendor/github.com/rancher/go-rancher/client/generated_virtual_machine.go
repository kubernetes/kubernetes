package client

const (
	VIRTUAL_MACHINE_TYPE = "virtualMachine"
)

type VirtualMachine struct {
	Resource

	AccountId string `json:"accountId,omitempty" yaml:"account_id,omitempty"`

	AgentId string `json:"agentId,omitempty" yaml:"agent_id,omitempty"`

	AllocationState string `json:"allocationState,omitempty" yaml:"allocation_state,omitempty"`

	BlkioDeviceOptions map[string]interface{} `json:"blkioDeviceOptions,omitempty" yaml:"blkio_device_options,omitempty"`

	Command []string `json:"command,omitempty" yaml:"command,omitempty"`

	Count int64 `json:"count,omitempty" yaml:"count,omitempty"`

	CpuSet string `json:"cpuSet,omitempty" yaml:"cpu_set,omitempty"`

	CpuShares int64 `json:"cpuShares,omitempty" yaml:"cpu_shares,omitempty"`

	CreateIndex int64 `json:"createIndex,omitempty" yaml:"create_index,omitempty"`

	Created string `json:"created,omitempty" yaml:"created,omitempty"`

	Data map[string]interface{} `json:"data,omitempty" yaml:"data,omitempty"`

	DeploymentUnitUuid string `json:"deploymentUnitUuid,omitempty" yaml:"deployment_unit_uuid,omitempty"`

	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	Disks []interface{} `json:"disks,omitempty" yaml:"disks,omitempty"`

	Dns []string `json:"dns,omitempty" yaml:"dns,omitempty"`

	DnsSearch []string `json:"dnsSearch,omitempty" yaml:"dns_search,omitempty"`

	DomainName string `json:"domainName,omitempty" yaml:"domain_name,omitempty"`

	Expose []string `json:"expose,omitempty" yaml:"expose,omitempty"`

	ExternalId string `json:"externalId,omitempty" yaml:"external_id,omitempty"`

	ExtraHosts []string `json:"extraHosts,omitempty" yaml:"extra_hosts,omitempty"`

	FirstRunning string `json:"firstRunning,omitempty" yaml:"first_running,omitempty"`

	HealthCheck *InstanceHealthCheck `json:"healthCheck,omitempty" yaml:"health_check,omitempty"`

	HealthState string `json:"healthState,omitempty" yaml:"health_state,omitempty"`

	HostId string `json:"hostId,omitempty" yaml:"host_id,omitempty"`

	Hostname string `json:"hostname,omitempty" yaml:"hostname,omitempty"`

	ImageUuid string `json:"imageUuid,omitempty" yaml:"image_uuid,omitempty"`

	InstanceLinks map[string]interface{} `json:"instanceLinks,omitempty" yaml:"instance_links,omitempty"`

	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`

	Labels map[string]interface{} `json:"labels,omitempty" yaml:"labels,omitempty"`

	LogConfig *LogConfig `json:"logConfig,omitempty" yaml:"log_config,omitempty"`

	Memory int64 `json:"memory,omitempty" yaml:"memory,omitempty"`

	MemoryMb int64 `json:"memoryMb,omitempty" yaml:"memory_mb,omitempty"`

	MemorySwap int64 `json:"memorySwap,omitempty" yaml:"memory_swap,omitempty"`

	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	NativeContainer bool `json:"nativeContainer,omitempty" yaml:"native_container,omitempty"`

	NetworkIds []string `json:"networkIds,omitempty" yaml:"network_ids,omitempty"`

	NetworkMode string `json:"networkMode,omitempty" yaml:"network_mode,omitempty"`

	Ports []string `json:"ports,omitempty" yaml:"ports,omitempty"`

	PrimaryIpAddress string `json:"primaryIpAddress,omitempty" yaml:"primary_ip_address,omitempty"`

	RegistryCredentialId string `json:"registryCredentialId,omitempty" yaml:"registry_credential_id,omitempty"`

	RemoveTime string `json:"removeTime,omitempty" yaml:"remove_time,omitempty"`

	Removed string `json:"removed,omitempty" yaml:"removed,omitempty"`

	RequestedHostId string `json:"requestedHostId,omitempty" yaml:"requested_host_id,omitempty"`

	RestartPolicy *RestartPolicy `json:"restartPolicy,omitempty" yaml:"restart_policy,omitempty"`

	SecurityOpt []string `json:"securityOpt,omitempty" yaml:"security_opt,omitempty"`

	StartCount int64 `json:"startCount,omitempty" yaml:"start_count,omitempty"`

	StartOnCreate bool `json:"startOnCreate,omitempty" yaml:"start_on_create,omitempty"`

	State string `json:"state,omitempty" yaml:"state,omitempty"`

	SystemContainer string `json:"systemContainer,omitempty" yaml:"system_container,omitempty"`

	Token string `json:"token,omitempty" yaml:"token,omitempty"`

	Transitioning string `json:"transitioning,omitempty" yaml:"transitioning,omitempty"`

	TransitioningMessage string `json:"transitioningMessage,omitempty" yaml:"transitioning_message,omitempty"`

	TransitioningProgress int64 `json:"transitioningProgress,omitempty" yaml:"transitioning_progress,omitempty"`

	Userdata string `json:"userdata,omitempty" yaml:"userdata,omitempty"`

	Uuid string `json:"uuid,omitempty" yaml:"uuid,omitempty"`

	Vcpu int64 `json:"vcpu,omitempty" yaml:"vcpu,omitempty"`

	Version string `json:"version,omitempty" yaml:"version,omitempty"`

	VolumeDriver string `json:"volumeDriver,omitempty" yaml:"volume_driver,omitempty"`
}

type VirtualMachineCollection struct {
	Collection
	Data []VirtualMachine `json:"data,omitempty"`
}

type VirtualMachineClient struct {
	rancherClient *RancherClient
}

type VirtualMachineOperations interface {
	List(opts *ListOpts) (*VirtualMachineCollection, error)
	Create(opts *VirtualMachine) (*VirtualMachine, error)
	Update(existing *VirtualMachine, updates interface{}) (*VirtualMachine, error)
	ById(id string) (*VirtualMachine, error)
	Delete(container *VirtualMachine) error

	ActionAllocate(*VirtualMachine) (*Instance, error)

	ActionConsole(*VirtualMachine, *InstanceConsoleInput) (*InstanceConsole, error)

	ActionCreate(*VirtualMachine) (*Instance, error)

	ActionDeallocate(*VirtualMachine) (*Instance, error)

	ActionError(*VirtualMachine) (*Instance, error)

	ActionExecute(*VirtualMachine, *ContainerExec) (*HostAccess, error)

	ActionLogs(*VirtualMachine, *ContainerLogs) (*HostAccess, error)

	ActionMigrate(*VirtualMachine) (*Instance, error)

	ActionProxy(*VirtualMachine, *ContainerProxy) (*HostAccess, error)

	ActionPurge(*VirtualMachine) (*Instance, error)

	ActionRemove(*VirtualMachine) (*Instance, error)

	ActionRestart(*VirtualMachine) (*Instance, error)

	ActionRestore(*VirtualMachine) (*Instance, error)

	ActionSetlabels(*VirtualMachine, *SetLabelsInput) (*Container, error)

	ActionStart(*VirtualMachine) (*Instance, error)

	ActionStop(*VirtualMachine, *InstanceStop) (*Instance, error)

	ActionUpdate(*VirtualMachine) (*Instance, error)

	ActionUpdatehealthy(*VirtualMachine) (*Instance, error)

	ActionUpdatereinitializing(*VirtualMachine) (*Instance, error)

	ActionUpdateunhealthy(*VirtualMachine) (*Instance, error)
}

func newVirtualMachineClient(rancherClient *RancherClient) *VirtualMachineClient {
	return &VirtualMachineClient{
		rancherClient: rancherClient,
	}
}

func (c *VirtualMachineClient) Create(container *VirtualMachine) (*VirtualMachine, error) {
	resp := &VirtualMachine{}
	err := c.rancherClient.doCreate(VIRTUAL_MACHINE_TYPE, container, resp)
	return resp, err
}

func (c *VirtualMachineClient) Update(existing *VirtualMachine, updates interface{}) (*VirtualMachine, error) {
	resp := &VirtualMachine{}
	err := c.rancherClient.doUpdate(VIRTUAL_MACHINE_TYPE, &existing.Resource, updates, resp)
	return resp, err
}

func (c *VirtualMachineClient) List(opts *ListOpts) (*VirtualMachineCollection, error) {
	resp := &VirtualMachineCollection{}
	err := c.rancherClient.doList(VIRTUAL_MACHINE_TYPE, opts, resp)
	return resp, err
}

func (c *VirtualMachineClient) ById(id string) (*VirtualMachine, error) {
	resp := &VirtualMachine{}
	err := c.rancherClient.doById(VIRTUAL_MACHINE_TYPE, id, resp)
	if apiError, ok := err.(*ApiError); ok {
		if apiError.StatusCode == 404 {
			return nil, nil
		}
	}
	return resp, err
}

func (c *VirtualMachineClient) Delete(container *VirtualMachine) error {
	return c.rancherClient.doResourceDelete(VIRTUAL_MACHINE_TYPE, &container.Resource)
}

func (c *VirtualMachineClient) ActionAllocate(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "allocate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionConsole(resource *VirtualMachine, input *InstanceConsoleInput) (*InstanceConsole, error) {

	resp := &InstanceConsole{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "console", &resource.Resource, input, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionCreate(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "create", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionDeallocate(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "deallocate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionError(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "error", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionExecute(resource *VirtualMachine, input *ContainerExec) (*HostAccess, error) {

	resp := &HostAccess{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "execute", &resource.Resource, input, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionLogs(resource *VirtualMachine, input *ContainerLogs) (*HostAccess, error) {

	resp := &HostAccess{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "logs", &resource.Resource, input, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionMigrate(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "migrate", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionProxy(resource *VirtualMachine, input *ContainerProxy) (*HostAccess, error) {

	resp := &HostAccess{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "proxy", &resource.Resource, input, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionPurge(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "purge", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionRemove(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "remove", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionRestart(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "restart", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionRestore(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "restore", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionSetlabels(resource *VirtualMachine, input *SetLabelsInput) (*Container, error) {

	resp := &Container{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "setlabels", &resource.Resource, input, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionStart(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "start", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionStop(resource *VirtualMachine, input *InstanceStop) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "stop", &resource.Resource, input, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionUpdate(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "update", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionUpdatehealthy(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "updatehealthy", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionUpdatereinitializing(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "updatereinitializing", &resource.Resource, nil, resp)

	return resp, err
}

func (c *VirtualMachineClient) ActionUpdateunhealthy(resource *VirtualMachine) (*Instance, error) {

	resp := &Instance{}

	err := c.rancherClient.doAction(VIRTUAL_MACHINE_TYPE, "updateunhealthy", &resource.Resource, nil, resp)

	return resp, err
}
