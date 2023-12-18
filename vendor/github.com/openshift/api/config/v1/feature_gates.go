package v1

// FeatureGateDescription is a golang-only interface used to contains details for a feature gate.
type FeatureGateDescription struct {
	// FeatureGateAttributes is the information that appears in the API
	FeatureGateAttributes FeatureGateAttributes

	// OwningJiraComponent is the jira component that owns most of the impl and first assignment for the bug.
	// This is the team that owns the feature long term.
	OwningJiraComponent string
	// ResponsiblePerson is the person who is on the hook for first contact.  This is often, but not always, a team lead.
	// It is someone who can make the promise on the behalf of the team.
	ResponsiblePerson string
	// OwningProduct is the product that owns the lifecycle of the gate.
	OwningProduct OwningProduct
}

type OwningProduct string

var (
	ocpSpecific = OwningProduct("OCP")
	kubernetes  = OwningProduct("Kubernetes")
)

var (
	FeatureGateValidatingAdmissionPolicy = FeatureGateName("ValidatingAdmissionPolicy")
	validatingAdmissionPolicy            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateValidatingAdmissionPolicy,
		},
		OwningJiraComponent: "kube-apiserver",
		ResponsiblePerson:   "benluddy",
		OwningProduct:       kubernetes,
	}

	FeatureGateGatewayAPI = FeatureGateName("GatewayAPI")
	gateGatewayAPI        = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateGatewayAPI,
		},
		OwningJiraComponent: "Routing",
		ResponsiblePerson:   "miciah",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateOpenShiftPodSecurityAdmission = FeatureGateName("OpenShiftPodSecurityAdmission")
	openShiftPodSecurityAdmission            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateOpenShiftPodSecurityAdmission,
		},
		OwningJiraComponent: "auth",
		ResponsiblePerson:   "stlaz",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateExternalCloudProvider = FeatureGateName("ExternalCloudProvider")
	externalCloudProvider            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateExternalCloudProvider,
		},
		OwningJiraComponent: "cloud-provider",
		ResponsiblePerson:   "jspeed",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateExternalCloudProviderAzure = FeatureGateName("ExternalCloudProviderAzure")
	externalCloudProviderAzure            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateExternalCloudProviderAzure,
		},
		OwningJiraComponent: "cloud-provider",
		ResponsiblePerson:   "jspeed",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateExternalCloudProviderGCP = FeatureGateName("ExternalCloudProviderGCP")
	externalCloudProviderGCP            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateExternalCloudProviderGCP,
		},
		OwningJiraComponent: "cloud-provider",
		ResponsiblePerson:   "jspeed",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateExternalCloudProviderExternal = FeatureGateName("ExternalCloudProviderExternal")
	externalCloudProviderExternal            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateExternalCloudProviderExternal,
		},
		OwningJiraComponent: "cloud-provider",
		ResponsiblePerson:   "elmiko",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateCSIDriverSharedResource = FeatureGateName("CSIDriverSharedResource")
	csiDriverSharedResource            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateCSIDriverSharedResource,
		},
		OwningJiraComponent: "builds",
		ResponsiblePerson:   "adkaplan",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateBuildCSIVolumes = FeatureGateName("BuildCSIVolumes")
	buildCSIVolumes            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateBuildCSIVolumes,
		},
		OwningJiraComponent: "builds",
		ResponsiblePerson:   "adkaplan",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateNodeSwap = FeatureGateName("NodeSwap")
	nodeSwap            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateNodeSwap,
		},
		OwningJiraComponent: "node",
		ResponsiblePerson:   "ehashman",
		OwningProduct:       kubernetes,
	}

	FeatureGateMachineAPIProviderOpenStack = FeatureGateName("MachineAPIProviderOpenStack")
	machineAPIProviderOpenStack            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateMachineAPIProviderOpenStack,
		},
		OwningJiraComponent: "openstack",
		ResponsiblePerson:   "egarcia",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateInsightsConfigAPI = FeatureGateName("InsightsConfigAPI")
	insightsConfigAPI            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateInsightsConfigAPI,
		},
		OwningJiraComponent: "insights",
		ResponsiblePerson:   "tremes",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateDynamicResourceAllocation = FeatureGateName("DynamicResourceAllocation")
	dynamicResourceAllocation            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateDynamicResourceAllocation,
		},
		OwningJiraComponent: "scheduling",
		ResponsiblePerson:   "jchaloup",
		OwningProduct:       kubernetes,
	}

	FeatureGateAzureWorkloadIdentity = FeatureGateName("AzureWorkloadIdentity")
	azureWorkloadIdentity            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateAzureWorkloadIdentity,
		},
		OwningJiraComponent: "cloud-credential-operator",
		ResponsiblePerson:   "abutcher",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateMaxUnavailableStatefulSet = FeatureGateName("MaxUnavailableStatefulSet")
	maxUnavailableStatefulSet            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateMaxUnavailableStatefulSet,
		},
		OwningJiraComponent: "apps",
		ResponsiblePerson:   "atiratree",
		OwningProduct:       kubernetes,
	}

	FeatureGateEventedPLEG = FeatureGateName("EventedPLEG")
	eventedPleg            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateEventedPLEG,
		},
		OwningJiraComponent: "node",
		ResponsiblePerson:   "sairameshv",
		OwningProduct:       kubernetes,
	}

	FeatureGatePrivateHostedZoneAWS = FeatureGateName("PrivateHostedZoneAWS")
	privateHostedZoneAWS            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGatePrivateHostedZoneAWS,
		},
		OwningJiraComponent: "Routing",
		ResponsiblePerson:   "miciah",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateSigstoreImageVerification = FeatureGateName("SigstoreImageVerification")
	sigstoreImageVerification            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateSigstoreImageVerification,
		},
		OwningJiraComponent: "node",
		ResponsiblePerson:   "sgrunert",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateGCPLabelsTags = FeatureGateName("GCPLabelsTags")
	gcpLabelsTags            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateGCPLabelsTags,
		},
		OwningJiraComponent: "Installer",
		ResponsiblePerson:   "bhb",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateAlibabaPlatform = FeatureGateName("AlibabaPlatform")
	alibabaPlatform            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateAlibabaPlatform,
		},
		OwningJiraComponent: "cloud-provider",
		ResponsiblePerson:   "jspeed",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateCloudDualStackNodeIPs = FeatureGateName("CloudDualStackNodeIPs")
	cloudDualStackNodeIPs            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateCloudDualStackNodeIPs,
		},
		OwningJiraComponent: "machine-config-operator/platform-baremetal",
		ResponsiblePerson:   "mkowalsk",
		OwningProduct:       kubernetes,
	}
	FeatureGateVSphereStaticIPs = FeatureGateName("VSphereStaticIPs")
	vSphereStaticIPs            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateVSphereStaticIPs,
		},
		OwningJiraComponent: "splat",
		ResponsiblePerson:   "rvanderp3",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateRouteExternalCertificate = FeatureGateName("RouteExternalCertificate")
	routeExternalCertificate            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateRouteExternalCertificate,
		},
		OwningJiraComponent: "router",
		ResponsiblePerson:   "thejasn",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateAdminNetworkPolicy = FeatureGateName("AdminNetworkPolicy")
	adminNetworkPolicy            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateAdminNetworkPolicy,
		},
		OwningJiraComponent: "Networking/ovn-kubernetes",
		ResponsiblePerson:   "tssurya",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateNetworkLiveMigration = FeatureGateName("NetworkLiveMigration")
	sdnLiveMigration                = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateNetworkLiveMigration,
		},
		OwningJiraComponent: "Networking/ovn-kubernetes",
		ResponsiblePerson:   "pliu",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateAutomatedEtcdBackup = FeatureGateName("AutomatedEtcdBackup")
	automatedEtcdBackup            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateAutomatedEtcdBackup,
		},
		OwningJiraComponent: "etcd",
		ResponsiblePerson:   "hasbro17",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateMachineAPIOperatorDisableMachineHealthCheckController = FeatureGateName("MachineAPIOperatorDisableMachineHealthCheckController")
	machineAPIOperatorDisableMachineHealthCheckController            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateMachineAPIOperatorDisableMachineHealthCheckController,
		},
		OwningJiraComponent: "ecoproject",
		ResponsiblePerson:   "msluiter",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateDNSNameResolver = FeatureGateName("DNSNameResolver")
	dnsNameResolver            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateDNSNameResolver,
		},
		OwningJiraComponent: "dns",
		ResponsiblePerson:   "miciah",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateVSphereControlPlaneMachineset = FeatureGateName("VSphereControlPlaneMachineSet")
	vSphereControlPlaneMachineset            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateVSphereControlPlaneMachineset,
		},
		OwningJiraComponent: "splat",
		ResponsiblePerson:   "rvanderp3",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateMachineConfigNodes = FeatureGateName("MachineConfigNodes")
	machineConfigNodes            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateMachineConfigNodes,
		},
		OwningJiraComponent: "MachineConfigOperator",
		ResponsiblePerson:   "cdoern",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateClusterAPIInstall = FeatureGateName("ClusterAPIInstall")
	clusterAPIInstall            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateClusterAPIInstall,
		},
		OwningJiraComponent: "Installer",
		ResponsiblePerson:   "vincepri",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateMetricsServer = FeatureGateName("MetricsServer")
	metricsServer            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateMetricsServer,
		},
		OwningJiraComponent: "Monitoring",
		ResponsiblePerson:   "slashpai",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateInstallAlternateInfrastructureAWS = FeatureGateName("InstallAlternateInfrastructureAWS")
	installAlternateInfrastructureAWS            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateInstallAlternateInfrastructureAWS,
		},
		OwningJiraComponent: "Installer",
		ResponsiblePerson:   "padillon",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateGCPClusterHostedDNS = FeatureGateName("GCPClusterHostedDNS")
	gcpClusterHostedDNS            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateGCPClusterHostedDNS,
		},
		OwningJiraComponent: "Installer",
		ResponsiblePerson:   "barbacbd",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateMixedCPUsAllocation = FeatureGateName("MixedCPUsAllocation")
	mixedCPUsAllocation            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateMixedCPUsAllocation,
		},
		OwningJiraComponent: "NodeTuningOperator",
		ResponsiblePerson:   "titzhak",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateManagedBootImages = FeatureGateName("ManagedBootImages")
	managedBootImages            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateManagedBootImages,
		},
		OwningJiraComponent: "MachineConfigOperator",
		ResponsiblePerson:   "djoshy",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateDisableKubeletCloudCredentialProviders = FeatureGateName("DisableKubeletCloudCredentialProviders")
	disableKubeletCloudCredentialProviders            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateDisableKubeletCloudCredentialProviders,
		},
		OwningJiraComponent: "cloud-provider",
		ResponsiblePerson:   "jspeed",
		OwningProduct:       kubernetes,
	}

	FeatureGateOnClusterBuild = FeatureGateName("OnClusterBuild")
	onClusterBuild            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateOnClusterBuild,
		},
		OwningJiraComponent: "MachineConfigOperator",
		ResponsiblePerson:   "dkhater",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateSignatureStores = FeatureGateName("SignatureStores")
	signatureStores            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateSignatureStores,
		},
		OwningJiraComponent: "over-the-air-updates",
		ResponsiblePerson:   "lmohanty",
		OwningProduct:       ocpSpecific,
	}

	FeatureGateKMSv1 = FeatureGateName("KMSv1")
	kmsv1            = FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: FeatureGateKMSv1,
		},
		OwningJiraComponent: "kube-apiserver",
		ResponsiblePerson:   "dgrisonnet",
		OwningProduct:       kubernetes,
	}
)
