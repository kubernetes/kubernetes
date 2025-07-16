package features

import "k8s.io/apimachinery/pkg/util/sets"

var legacyFeatureGates = sets.New(
	"AWSClusterHostedDNS",
	// never add to this list, if you think you have an exception ask @deads2k
	"AWSEFSDriverVolumeMetrics",
	// never add to this list, if you think you have an exception ask @deads2k
	"AdditionalRoutingCapabilities",
	// never add to this list, if you think you have an exception ask @deads2k
	"AdminNetworkPolicy",
	// never add to this list, if you think you have an exception ask @deads2k
	"AlibabaPlatform",
	// never add to this list, if you think you have an exception ask @deads2k
	"AutomatedEtcdBackup",
	// never add to this list, if you think you have an exception ask @deads2k
	"AzureWorkloadIdentity",
	// never add to this list, if you think you have an exception ask @deads2k
	"BootcNodeManagement",
	// never add to this list, if you think you have an exception ask @deads2k
	"BuildCSIVolumes",
	// never add to this list, if you think you have an exception ask @deads2k
	"ChunkSizeMiB",
	// never add to this list, if you think you have an exception ask @deads2k
	"ClusterAPIInstall",
	// never add to this list, if you think you have an exception ask @deads2k
	"ClusterAPIInstallIBMCloud",
	// never add to this list, if you think you have an exception ask @deads2k
	"ClusterMonitoringConfig",
	// never add to this list, if you think you have an exception ask @deads2k
	"DNSNameResolver",
	// never add to this list, if you think you have an exception ask @deads2k
	"EtcdBackendQuota",
	// never add to this list, if you think you have an exception ask @deads2k
	"Example",
	// never add to this list, if you think you have an exception ask @deads2k
	"Example2",
	// never add to this list, if you think you have an exception ask @deads2k
	"GCPClusterHostedDNS",
	// never add to this list, if you think you have an exception ask @deads2k
	"GCPLabelsTags",
	// never add to this list, if you think you have an exception ask @deads2k
	"GatewayAPI",
	// never add to this list, if you think you have an exception ask @deads2k
	"HardwareSpeed",
	// never add to this list, if you think you have an exception ask @deads2k
	"ImageStreamImportMode",
	// never add to this list, if you think you have an exception ask @deads2k
	"IngressControllerDynamicConfigurationManager",
	// never add to this list, if you think you have an exception ask @deads2k
	"IngressControllerLBSubnetsAWS",
	// never add to this list, if you think you have an exception ask @deads2k
	"InsightsConfig",
	// never add to this list, if you think you have an exception ask @deads2k
	"InsightsConfigAPI",
	// never add to this list, if you think you have an exception ask @deads2k
	"InsightsOnDemandDataGather",
	// never add to this list, if you think you have an exception ask @deads2k
	"InsightsRuntimeExtractor",
	// never add to this list, if you think you have an exception ask @deads2k
	"KMSv1",
	// never add to this list, if you think you have an exception ask @deads2k
	"MachineAPIMigration",
	// never add to this list, if you think you have an exception ask @deads2k
	"MachineAPIOperatorDisableMachineHealthCheckController",
	// never add to this list, if you think you have an exception ask @deads2k
	"MachineAPIProviderOpenStack",
	// never add to this list, if you think you have an exception ask @deads2k
	"MachineConfigNodes",
	// never add to this list, if you think you have an exception ask @deads2k
	"ManagedBootImages",
	// never add to this list, if you think you have an exception ask @deads2k
	"ManagedBootImagesAWS",
	// never add to this list, if you think you have an exception ask @deads2k
	"MetricsCollectionProfiles",
	// never add to this list, if you think you have an exception ask @deads2k
	"MixedCPUsAllocation",
	// never add to this list, if you think you have an exception ask @deads2k
	"MultiArchInstallAWS",
	// never add to this list, if you think you have an exception ask @deads2k
	"MultiArchInstallAzure",
	// never add to this list, if you think you have an exception ask @deads2k
	"MultiArchInstallGCP",
	// never add to this list, if you think you have an exception ask @deads2k
	"NetworkDiagnosticsConfig",
	// never add to this list, if you think you have an exception ask @deads2k
	"NetworkLiveMigration",
	// never add to this list, if you think you have an exception ask @deads2k
	"NetworkSegmentation",
	// never add to this list, if you think you have an exception ask @deads2k
	"NewOLM",
	// never add to this list, if you think you have an exception ask @deads2k
	"OVNObservability",
	// never add to this list, if you think you have an exception ask @deads2k
	"OnClusterBuild",
	// never add to this list, if you think you have an exception ask @deads2k
	"PersistentIPsForVirtualization",
	// never add to this list, if you think you have an exception ask @deads2k
	"PinnedImages",
	// never add to this list, if you think you have an exception ask @deads2k
	"PrivateHostedZoneAWS",
	// never add to this list, if you think you have an exception ask @deads2k
	"RouteAdvertisements",
	// never add to this list, if you think you have an exception ask @deads2k
	"RouteExternalCertificate",
	// never add to this list, if you think you have an exception ask @deads2k
	"SetEIPForNLBIngressController",
	// never add to this list, if you think you have an exception ask @deads2k
	"SignatureStores",
	// never add to this list, if you think you have an exception ask @deads2k
	"SigstoreImageVerification",
	// never add to this list, if you think you have an exception ask @deads2k
	"UpgradeStatus",
	// never add to this list, if you think you have an exception ask @deads2k
	"VSphereControlPlaneMachineSet",
	// never add to this list, if you think you have an exception ask @deads2k
	"VSphereDriverConfiguration",
	// never add to this list, if you think you have an exception ask @deads2k
	"VSphereMultiNetworks",
	// never add to this list, if you think you have an exception ask @deads2k
	"VSphereMultiVCenters",
	// never add to this list, if you think you have an exception ask @deads2k
	"VSphereStaticIPs",
)
