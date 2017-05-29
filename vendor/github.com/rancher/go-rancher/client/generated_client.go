package client

type RancherClient struct {
	RancherBaseClient

	Account                                  AccountOperations
	ActiveSetting                            ActiveSettingOperations
	AddOutputsInput                          AddOutputsInputOperations
	AddRemoveLoadBalancerServiceLinkInput    AddRemoveLoadBalancerServiceLinkInputOperations
	AddRemoveServiceLinkInput                AddRemoveServiceLinkInputOperations
	Agent                                    AgentOperations
	Amazonec2Config                          Amazonec2ConfigOperations
	ApiKey                                   ApiKeyOperations
	AuditLog                                 AuditLogOperations
	AzureConfig                              AzureConfigOperations
	Azureadconfig                            AzureadconfigOperations
	Backup                                   BackupOperations
	BackupTarget                             BackupTargetOperations
	BaseMachineConfig                        BaseMachineConfigOperations
	BlkioDeviceOption                        BlkioDeviceOptionOperations
	Certificate                              CertificateOperations
	ChangeSecretInput                        ChangeSecretInputOperations
	ComposeConfig                            ComposeConfigOperations
	ComposeConfigInput                       ComposeConfigInputOperations
	ComposeProject                           ComposeProjectOperations
	ComposeService                           ComposeServiceOperations
	ConfigItem                               ConfigItemOperations
	ConfigItemStatus                         ConfigItemStatusOperations
	Container                                ContainerOperations
	ContainerEvent                           ContainerEventOperations
	ContainerExec                            ContainerExecOperations
	ContainerLogs                            ContainerLogsOperations
	ContainerProxy                           ContainerProxyOperations
	Credential                               CredentialOperations
	Databasechangelog                        DatabasechangelogOperations
	Databasechangeloglock                    DatabasechangeloglockOperations
	DigitaloceanConfig                       DigitaloceanConfigOperations
	DnsService                               DnsServiceOperations
	DockerBuild                              DockerBuildOperations
	DynamicSchema                            DynamicSchemaOperations
	Environment                              EnvironmentOperations
	EnvironmentUpgrade                       EnvironmentUpgradeOperations
	ExtensionImplementation                  ExtensionImplementationOperations
	ExtensionPoint                           ExtensionPointOperations
	ExternalDnsEvent                         ExternalDnsEventOperations
	ExternalEvent                            ExternalEventOperations
	ExternalHandler                          ExternalHandlerOperations
	ExternalHandlerExternalHandlerProcessMap ExternalHandlerExternalHandlerProcessMapOperations
	ExternalHandlerProcess                   ExternalHandlerProcessOperations
	ExternalHandlerProcessConfig             ExternalHandlerProcessConfigOperations
	ExternalHostEvent                        ExternalHostEventOperations
	ExternalService                          ExternalServiceOperations
	ExternalServiceEvent                     ExternalServiceEventOperations
	ExternalStoragePoolEvent                 ExternalStoragePoolEventOperations
	ExternalVolumeEvent                      ExternalVolumeEventOperations
	FieldDocumentation                       FieldDocumentationOperations
	Githubconfig                             GithubconfigOperations
	HaConfig                                 HaConfigOperations
	HaConfigInput                            HaConfigInputOperations
	HaproxyConfig                            HaproxyConfigOperations
	HealthcheckInstanceHostMap               HealthcheckInstanceHostMapOperations
	Host                                     HostOperations
	HostAccess                               HostAccessOperations
	HostApiProxyToken                        HostApiProxyTokenOperations
	Identity                                 IdentityOperations
	Image                                    ImageOperations
	InServiceUpgradeStrategy                 InServiceUpgradeStrategyOperations
	Instance                                 InstanceOperations
	InstanceConsole                          InstanceConsoleOperations
	InstanceConsoleInput                     InstanceConsoleInputOperations
	InstanceHealthCheck                      InstanceHealthCheckOperations
	InstanceLink                             InstanceLinkOperations
	InstanceStop                             InstanceStopOperations
	IpAddress                                IpAddressOperations
	IpAddressAssociateInput                  IpAddressAssociateInputOperations
	KubernetesService                        KubernetesServiceOperations
	KubernetesStack                          KubernetesStackOperations
	KubernetesStackUpgrade                   KubernetesStackUpgradeOperations
	Label                                    LabelOperations
	LaunchConfig                             LaunchConfigOperations
	Ldapconfig                               LdapconfigOperations
	LoadBalancerAppCookieStickinessPolicy    LoadBalancerAppCookieStickinessPolicyOperations
	LoadBalancerConfig                       LoadBalancerConfigOperations
	LoadBalancerCookieStickinessPolicy       LoadBalancerCookieStickinessPolicyOperations
	LoadBalancerService                      LoadBalancerServiceOperations
	LoadBalancerServiceLink                  LoadBalancerServiceLinkOperations
	LocalAuthConfig                          LocalAuthConfigOperations
	LogConfig                                LogConfigOperations
	Machine                                  MachineOperations
	MachineDriver                            MachineDriverOperations
	Mount                                    MountOperations
	Network                                  NetworkOperations
	NfsConfig                                NfsConfigOperations
	Openldapconfig                           OpenldapconfigOperations
	PacketConfig                             PacketConfigOperations
	Password                                 PasswordOperations
	PhysicalHost                             PhysicalHostOperations
	Port                                     PortOperations
	ProcessDefinition                        ProcessDefinitionOperations
	ProcessExecution                         ProcessExecutionOperations
	ProcessInstance                          ProcessInstanceOperations
	Project                                  ProjectOperations
	ProjectMember                            ProjectMemberOperations
	PublicEndpoint                           PublicEndpointOperations
	Publish                                  PublishOperations
	PullTask                                 PullTaskOperations
	RecreateOnQuorumStrategyConfig           RecreateOnQuorumStrategyConfigOperations
	Register                                 RegisterOperations
	RegistrationToken                        RegistrationTokenOperations
	Registry                                 RegistryOperations
	RegistryCredential                       RegistryCredentialOperations
	ResourceDefinition                       ResourceDefinitionOperations
	RestartPolicy                            RestartPolicyOperations
	RestoreFromBackupInput                   RestoreFromBackupInputOperations
	RevertToSnapshotInput                    RevertToSnapshotInputOperations
	RollingRestartStrategy                   RollingRestartStrategyOperations
	ScalePolicy                              ScalePolicyOperations
	SecondaryLaunchConfig                    SecondaryLaunchConfigOperations
	Service                                  ServiceOperations
	ServiceConsumeMap                        ServiceConsumeMapOperations
	ServiceEvent                             ServiceEventOperations
	ServiceExposeMap                         ServiceExposeMapOperations
	ServiceLink                              ServiceLinkOperations
	ServiceProxy                             ServiceProxyOperations
	ServiceRestart                           ServiceRestartOperations
	ServiceUpgrade                           ServiceUpgradeOperations
	ServiceUpgradeStrategy                   ServiceUpgradeStrategyOperations
	ServicesPortRange                        ServicesPortRangeOperations
	SetLabelsInput                           SetLabelsInputOperations
	SetLoadBalancerServiceLinksInput         SetLoadBalancerServiceLinksInputOperations
	SetProjectMembersInput                   SetProjectMembersInputOperations
	SetServiceLinksInput                     SetServiceLinksInputOperations
	Setting                                  SettingOperations
	Snapshot                                 SnapshotOperations
	SnapshotBackupInput                      SnapshotBackupInputOperations
	StateTransition                          StateTransitionOperations
	StatsAccess                              StatsAccessOperations
	StoragePool                              StoragePoolOperations
	Subscribe                                SubscribeOperations
	Task                                     TaskOperations
	TaskInstance                             TaskInstanceOperations
	ToServiceUpgradeStrategy                 ToServiceUpgradeStrategyOperations
	TypeDocumentation                        TypeDocumentationOperations
	VirtualMachine                           VirtualMachineOperations
	VirtualMachineDisk                       VirtualMachineDiskOperations
	Volume                                   VolumeOperations
	VolumeSnapshotInput                      VolumeSnapshotInputOperations
}

func constructClient(rancherBaseClient *RancherBaseClientImpl) *RancherClient {
	client := &RancherClient{
		RancherBaseClient: rancherBaseClient,
	}

	client.Account = newAccountClient(client)
	client.ActiveSetting = newActiveSettingClient(client)
	client.AddOutputsInput = newAddOutputsInputClient(client)
	client.AddRemoveLoadBalancerServiceLinkInput = newAddRemoveLoadBalancerServiceLinkInputClient(client)
	client.AddRemoveServiceLinkInput = newAddRemoveServiceLinkInputClient(client)
	client.Agent = newAgentClient(client)
	client.Amazonec2Config = newAmazonec2ConfigClient(client)
	client.ApiKey = newApiKeyClient(client)
	client.AuditLog = newAuditLogClient(client)
	client.AzureConfig = newAzureConfigClient(client)
	client.Azureadconfig = newAzureadconfigClient(client)
	client.Backup = newBackupClient(client)
	client.BackupTarget = newBackupTargetClient(client)
	client.BaseMachineConfig = newBaseMachineConfigClient(client)
	client.BlkioDeviceOption = newBlkioDeviceOptionClient(client)
	client.Certificate = newCertificateClient(client)
	client.ChangeSecretInput = newChangeSecretInputClient(client)
	client.ComposeConfig = newComposeConfigClient(client)
	client.ComposeConfigInput = newComposeConfigInputClient(client)
	client.ComposeProject = newComposeProjectClient(client)
	client.ComposeService = newComposeServiceClient(client)
	client.ConfigItem = newConfigItemClient(client)
	client.ConfigItemStatus = newConfigItemStatusClient(client)
	client.Container = newContainerClient(client)
	client.ContainerEvent = newContainerEventClient(client)
	client.ContainerExec = newContainerExecClient(client)
	client.ContainerLogs = newContainerLogsClient(client)
	client.ContainerProxy = newContainerProxyClient(client)
	client.Credential = newCredentialClient(client)
	client.Databasechangelog = newDatabasechangelogClient(client)
	client.Databasechangeloglock = newDatabasechangeloglockClient(client)
	client.DigitaloceanConfig = newDigitaloceanConfigClient(client)
	client.DnsService = newDnsServiceClient(client)
	client.DockerBuild = newDockerBuildClient(client)
	client.DynamicSchema = newDynamicSchemaClient(client)
	client.Environment = newEnvironmentClient(client)
	client.EnvironmentUpgrade = newEnvironmentUpgradeClient(client)
	client.ExtensionImplementation = newExtensionImplementationClient(client)
	client.ExtensionPoint = newExtensionPointClient(client)
	client.ExternalDnsEvent = newExternalDnsEventClient(client)
	client.ExternalEvent = newExternalEventClient(client)
	client.ExternalHandler = newExternalHandlerClient(client)
	client.ExternalHandlerExternalHandlerProcessMap = newExternalHandlerExternalHandlerProcessMapClient(client)
	client.ExternalHandlerProcess = newExternalHandlerProcessClient(client)
	client.ExternalHandlerProcessConfig = newExternalHandlerProcessConfigClient(client)
	client.ExternalHostEvent = newExternalHostEventClient(client)
	client.ExternalService = newExternalServiceClient(client)
	client.ExternalServiceEvent = newExternalServiceEventClient(client)
	client.ExternalStoragePoolEvent = newExternalStoragePoolEventClient(client)
	client.ExternalVolumeEvent = newExternalVolumeEventClient(client)
	client.FieldDocumentation = newFieldDocumentationClient(client)
	client.Githubconfig = newGithubconfigClient(client)
	client.HaConfig = newHaConfigClient(client)
	client.HaConfigInput = newHaConfigInputClient(client)
	client.HaproxyConfig = newHaproxyConfigClient(client)
	client.HealthcheckInstanceHostMap = newHealthcheckInstanceHostMapClient(client)
	client.Host = newHostClient(client)
	client.HostAccess = newHostAccessClient(client)
	client.HostApiProxyToken = newHostApiProxyTokenClient(client)
	client.Identity = newIdentityClient(client)
	client.Image = newImageClient(client)
	client.InServiceUpgradeStrategy = newInServiceUpgradeStrategyClient(client)
	client.Instance = newInstanceClient(client)
	client.InstanceConsole = newInstanceConsoleClient(client)
	client.InstanceConsoleInput = newInstanceConsoleInputClient(client)
	client.InstanceHealthCheck = newInstanceHealthCheckClient(client)
	client.InstanceLink = newInstanceLinkClient(client)
	client.InstanceStop = newInstanceStopClient(client)
	client.IpAddress = newIpAddressClient(client)
	client.IpAddressAssociateInput = newIpAddressAssociateInputClient(client)
	client.KubernetesService = newKubernetesServiceClient(client)
	client.KubernetesStack = newKubernetesStackClient(client)
	client.KubernetesStackUpgrade = newKubernetesStackUpgradeClient(client)
	client.Label = newLabelClient(client)
	client.LaunchConfig = newLaunchConfigClient(client)
	client.Ldapconfig = newLdapconfigClient(client)
	client.LoadBalancerAppCookieStickinessPolicy = newLoadBalancerAppCookieStickinessPolicyClient(client)
	client.LoadBalancerConfig = newLoadBalancerConfigClient(client)
	client.LoadBalancerCookieStickinessPolicy = newLoadBalancerCookieStickinessPolicyClient(client)
	client.LoadBalancerService = newLoadBalancerServiceClient(client)
	client.LoadBalancerServiceLink = newLoadBalancerServiceLinkClient(client)
	client.LocalAuthConfig = newLocalAuthConfigClient(client)
	client.LogConfig = newLogConfigClient(client)
	client.Machine = newMachineClient(client)
	client.MachineDriver = newMachineDriverClient(client)
	client.Mount = newMountClient(client)
	client.Network = newNetworkClient(client)
	client.NfsConfig = newNfsConfigClient(client)
	client.Openldapconfig = newOpenldapconfigClient(client)
	client.PacketConfig = newPacketConfigClient(client)
	client.Password = newPasswordClient(client)
	client.PhysicalHost = newPhysicalHostClient(client)
	client.Port = newPortClient(client)
	client.ProcessDefinition = newProcessDefinitionClient(client)
	client.ProcessExecution = newProcessExecutionClient(client)
	client.ProcessInstance = newProcessInstanceClient(client)
	client.Project = newProjectClient(client)
	client.ProjectMember = newProjectMemberClient(client)
	client.PublicEndpoint = newPublicEndpointClient(client)
	client.Publish = newPublishClient(client)
	client.PullTask = newPullTaskClient(client)
	client.RecreateOnQuorumStrategyConfig = newRecreateOnQuorumStrategyConfigClient(client)
	client.Register = newRegisterClient(client)
	client.RegistrationToken = newRegistrationTokenClient(client)
	client.Registry = newRegistryClient(client)
	client.RegistryCredential = newRegistryCredentialClient(client)
	client.ResourceDefinition = newResourceDefinitionClient(client)
	client.RestartPolicy = newRestartPolicyClient(client)
	client.RestoreFromBackupInput = newRestoreFromBackupInputClient(client)
	client.RevertToSnapshotInput = newRevertToSnapshotInputClient(client)
	client.RollingRestartStrategy = newRollingRestartStrategyClient(client)
	client.ScalePolicy = newScalePolicyClient(client)
	client.SecondaryLaunchConfig = newSecondaryLaunchConfigClient(client)
	client.Service = newServiceClient(client)
	client.ServiceConsumeMap = newServiceConsumeMapClient(client)
	client.ServiceEvent = newServiceEventClient(client)
	client.ServiceExposeMap = newServiceExposeMapClient(client)
	client.ServiceLink = newServiceLinkClient(client)
	client.ServiceProxy = newServiceProxyClient(client)
	client.ServiceRestart = newServiceRestartClient(client)
	client.ServiceUpgrade = newServiceUpgradeClient(client)
	client.ServiceUpgradeStrategy = newServiceUpgradeStrategyClient(client)
	client.ServicesPortRange = newServicesPortRangeClient(client)
	client.SetLabelsInput = newSetLabelsInputClient(client)
	client.SetLoadBalancerServiceLinksInput = newSetLoadBalancerServiceLinksInputClient(client)
	client.SetProjectMembersInput = newSetProjectMembersInputClient(client)
	client.SetServiceLinksInput = newSetServiceLinksInputClient(client)
	client.Setting = newSettingClient(client)
	client.Snapshot = newSnapshotClient(client)
	client.SnapshotBackupInput = newSnapshotBackupInputClient(client)
	client.StateTransition = newStateTransitionClient(client)
	client.StatsAccess = newStatsAccessClient(client)
	client.StoragePool = newStoragePoolClient(client)
	client.Subscribe = newSubscribeClient(client)
	client.Task = newTaskClient(client)
	client.TaskInstance = newTaskInstanceClient(client)
	client.ToServiceUpgradeStrategy = newToServiceUpgradeStrategyClient(client)
	client.TypeDocumentation = newTypeDocumentationClient(client)
	client.VirtualMachine = newVirtualMachineClient(client)
	client.VirtualMachineDisk = newVirtualMachineDiskClient(client)
	client.Volume = newVolumeClient(client)
	client.VolumeSnapshotInput = newVolumeSnapshotInputClient(client)

	return client
}

func NewRancherClient(opts *ClientOpts) (*RancherClient, error) {
	rancherBaseClient := &RancherBaseClientImpl{
		Types: map[string]Schema{},
	}
	client := constructClient(rancherBaseClient)

	err := setupRancherBaseClient(rancherBaseClient, opts)
	if err != nil {
		return nil, err
	}

	return client, nil
}
