package features

import (
	"fmt"

	configv1 "github.com/openshift/api/config/v1"
)

func FeatureSets(clusterProfile ClusterProfileName, featureSet configv1.FeatureSet) (*FeatureGateEnabledDisabled, error) {
	byFeatureSet, ok := allFeatureGates[clusterProfile]
	if !ok {
		return nil, fmt.Errorf("no information found for ClusterProfile=%q", clusterProfile)
	}
	featureGates, ok := byFeatureSet[featureSet]
	if !ok {
		return nil, fmt.Errorf("no information found for FeatureSet=%q under ClusterProfile=%q", featureSet, clusterProfile)
	}
	return featureGates.DeepCopy(), nil
}

func AllFeatureSets() map[ClusterProfileName]map[configv1.FeatureSet]*FeatureGateEnabledDisabled {
	ret := map[ClusterProfileName]map[configv1.FeatureSet]*FeatureGateEnabledDisabled{}

	for clusterProfile, byFeatureSet := range allFeatureGates {
		newByFeatureSet := map[configv1.FeatureSet]*FeatureGateEnabledDisabled{}

		for featureSet, enabledDisabled := range byFeatureSet {
			newByFeatureSet[featureSet] = enabledDisabled.DeepCopy()
		}
		ret[clusterProfile] = newByFeatureSet
	}

	return ret
}

var (
	allFeatureGates = map[ClusterProfileName]map[configv1.FeatureSet]*FeatureGateEnabledDisabled{}

	FeatureGateConsolePluginCSP = newFeatureGate("ConsolePluginContentSecurityPolicy").
					reportProblemsToJiraComponent("Management Console").
					contactPerson("jhadvig").
					productScope(ocpSpecific).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					enhancementPR("https://github.com/openshift/enhancements/pull/1706").
					mustRegister()

	FeatureGateServiceAccountTokenNodeBinding = newFeatureGate("ServiceAccountTokenNodeBinding").
							reportProblemsToJiraComponent("apiserver-auth").
							contactPerson("stlaz").
							productScope(kubernetes).
							enhancementPR("https://github.com/kubernetes/enhancements/issues/4193").
							enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateValidatingAdmissionPolicy = newFeatureGate("ValidatingAdmissionPolicy").
						reportProblemsToJiraComponent("kube-apiserver").
						contactPerson("benluddy").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/3488").
						enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateGatewayAPI = newFeatureGate("GatewayAPI").
				reportProblemsToJiraComponent("Routing").
				contactPerson("miciah").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enableIn(configv1.DevPreviewNoUpgrade).
				mustRegister()

	FeatureGateSetEIPForNLBIngressController = newFeatureGate("SetEIPForNLBIngressController").
							reportProblemsToJiraComponent("Networking / router").
							contactPerson("miheer").
							productScope(ocpSpecific).
							enhancementPR(legacyFeatureGateWithoutEnhancement).
							enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateOpenShiftPodSecurityAdmission = newFeatureGate("OpenShiftPodSecurityAdmission").
							reportProblemsToJiraComponent("auth").
							contactPerson("ibihim").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/899").
							enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateBuildCSIVolumes = newFeatureGate("BuildCSIVolumes").
					reportProblemsToJiraComponent("builds").
					contactPerson("adkaplan").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNodeSwap = newFeatureGate("NodeSwap").
				reportProblemsToJiraComponent("node").
				contactPerson("ehashman").
				productScope(kubernetes).
				enhancementPR("https://github.com/kubernetes/enhancements/issues/2400").
				enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateMachineAPIProviderOpenStack = newFeatureGate("MachineAPIProviderOpenStack").
						reportProblemsToJiraComponent("openstack").
						contactPerson("egarcia").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateInsightsConfigAPI = newFeatureGate("InsightsConfigAPI").
					reportProblemsToJiraComponent("insights").
					contactPerson("tremes").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateInsightsRuntimeExtractor = newFeatureGate("InsightsRuntimeExtractor").
						reportProblemsToJiraComponent("insights").
						contactPerson("jmesnil").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateDynamicResourceAllocation = newFeatureGate("DynamicResourceAllocation").
						reportProblemsToJiraComponent("scheduling").
						contactPerson("jchaloup").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/4381").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateAzureWorkloadIdentity = newFeatureGate("AzureWorkloadIdentity").
						reportProblemsToJiraComponent("cloud-credential-operator").
						contactPerson("abutcher").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateMaxUnavailableStatefulSet = newFeatureGate("MaxUnavailableStatefulSet").
						reportProblemsToJiraComponent("apps").
						contactPerson("atiratree").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/961").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateEventedPLEG = newFeatureGate("EventedPLEG").
				reportProblemsToJiraComponent("node").
				contactPerson("sairameshv").
				productScope(kubernetes).
				enhancementPR("https://github.com/kubernetes/enhancements/issues/3386").
				mustRegister()

	FeatureGatePrivateHostedZoneAWS = newFeatureGate("PrivateHostedZoneAWS").
					reportProblemsToJiraComponent("Routing").
					contactPerson("miciah").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateSigstoreImageVerification = newFeatureGate("SigstoreImageVerification").
						reportProblemsToJiraComponent("node").
						contactPerson("sgrunert").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateGCPLabelsTags = newFeatureGate("GCPLabelsTags").
					reportProblemsToJiraComponent("Installer").
					contactPerson("bhb").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateAlibabaPlatform = newFeatureGate("AlibabaPlatform").
					reportProblemsToJiraComponent("cloud-provider").
					contactPerson("jspeed").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateCloudDualStackNodeIPs = newFeatureGate("CloudDualStackNodeIPs").
						reportProblemsToJiraComponent("machine-config-operator/platform-baremetal").
						contactPerson("mkowalsk").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/3705").
						enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateVSphereHostVMGroupZonal = newFeatureGate("VSphereHostVMGroupZonal").
						reportProblemsToJiraComponent("splat").
						contactPerson("jcpowermac").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1677").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateVSphereMultiDisk = newFeatureGate("VSphereMultiDisk").
					reportProblemsToJiraComponent("splat").
					contactPerson("vr4manta").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1709").
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateVSphereMultiVCenters = newFeatureGate("VSphereMultiVCenters").
					reportProblemsToJiraComponent("splat").
					contactPerson("vr4manta").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateVSphereStaticIPs = newFeatureGate("VSphereStaticIPs").
					reportProblemsToJiraComponent("splat").
					contactPerson("rvanderp3").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateRouteExternalCertificate = newFeatureGate("RouteExternalCertificate").
						reportProblemsToJiraComponent("router").
						contactPerson("thejasn").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateCPMSMachineNamePrefix = newFeatureGate("CPMSMachineNamePrefix").
						reportProblemsToJiraComponent("Cloud Compute / ControlPlaneMachineSet").
						contactPerson("chiragkyal").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1714").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateAdminNetworkPolicy = newFeatureGate("AdminNetworkPolicy").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("tssurya").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNetworkSegmentation = newFeatureGate("NetworkSegmentation").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("tssurya").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateAdditionalRoutingCapabilities = newFeatureGate("AdditionalRoutingCapabilities").
							reportProblemsToJiraComponent("Networking/cluster-network-operator").
							contactPerson("jcaamano").
							productScope(ocpSpecific).
							enhancementPR(legacyFeatureGateWithoutEnhancement).
							enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateRouteAdvertisements = newFeatureGate("RouteAdvertisements").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("jcaamano").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNetworkLiveMigration = newFeatureGate("NetworkLiveMigration").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("pliu").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNetworkDiagnosticsConfig = newFeatureGate("NetworkDiagnosticsConfig").
						reportProblemsToJiraComponent("Networking/cluster-network-operator").
						contactPerson("kyrtapz").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateOVNObservability = newFeatureGate("OVNObservability").
					reportProblemsToJiraComponent("Networking").
					contactPerson("npinaeva").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateHardwareSpeed = newFeatureGate("HardwareSpeed").
					reportProblemsToJiraComponent("etcd").
					contactPerson("hasbro17").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateBackendQuotaGiB = newFeatureGate("EtcdBackendQuota").
					reportProblemsToJiraComponent("etcd").
					contactPerson("hasbro17").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateAutomatedEtcdBackup = newFeatureGate("AutomatedEtcdBackup").
					reportProblemsToJiraComponent("etcd").
					contactPerson("hasbro17").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateMachineAPIOperatorDisableMachineHealthCheckController = newFeatureGate("MachineAPIOperatorDisableMachineHealthCheckController").
										reportProblemsToJiraComponent("ecoproject").
										contactPerson("msluiter").
										productScope(ocpSpecific).
										enhancementPR(legacyFeatureGateWithoutEnhancement).
										mustRegister()

	FeatureGateDNSNameResolver = newFeatureGate("DNSNameResolver").
					reportProblemsToJiraComponent("dns").
					contactPerson("miciah").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateVSphereControlPlaneMachineset = newFeatureGate("VSphereControlPlaneMachineSet").
							reportProblemsToJiraComponent("splat").
							contactPerson("rvanderp3").
							productScope(ocpSpecific).
							enhancementPR(legacyFeatureGateWithoutEnhancement).
							enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateMachineConfigNodes = newFeatureGate("MachineConfigNodes").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("cdoern").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateClusterAPIInstall = newFeatureGate("ClusterAPIInstall").
					reportProblemsToJiraComponent("Installer").
					contactPerson("vincepri").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					mustRegister()

	FeatureGateGCPClusterHostedDNS = newFeatureGate("GCPClusterHostedDNS").
					reportProblemsToJiraComponent("Installer").
					contactPerson("barbacbd").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateAWSClusterHostedDNS = newFeatureGate("AWSClusterHostedDNS").
					reportProblemsToJiraComponent("Installer").
					contactPerson("barbacbd").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateMixedCPUsAllocation = newFeatureGate("MixedCPUsAllocation").
					reportProblemsToJiraComponent("NodeTuningOperator").
					contactPerson("titzhak").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateManagedBootImages = newFeatureGate("ManagedBootImages").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("djoshy").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateManagedBootImagesAWS = newFeatureGate("ManagedBootImagesAWS").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("djoshy").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateDisableKubeletCloudCredentialProviders = newFeatureGate("DisableKubeletCloudCredentialProviders").
								reportProblemsToJiraComponent("cloud-provider").
								contactPerson("jspeed").
								productScope(kubernetes).
								enhancementPR("https://github.com/kubernetes/enhancements/issues/2395").
								enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
								mustRegister()

	FeatureGateOnClusterBuild = newFeatureGate("OnClusterBuild").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("dkhater").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateBootcNodeManagement = newFeatureGate("BootcNodeManagement").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("inesqyx").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateSignatureStores = newFeatureGate("SignatureStores").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("lmohanty").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateKMSv1 = newFeatureGate("KMSv1").
				reportProblemsToJiraComponent("kube-apiserver").
				contactPerson("dgrisonnet").
				productScope(kubernetes).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
				mustRegister()

	FeatureGatePinnedImages = newFeatureGate("PinnedImages").
				reportProblemsToJiraComponent("MachineConfigOperator").
				contactPerson("jhernand").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateUpgradeStatus = newFeatureGate("UpgradeStatus").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("pmuller").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateTranslateStreamCloseWebsocketRequests = newFeatureGate("TranslateStreamCloseWebsocketRequests").
								reportProblemsToJiraComponent("kube-apiserver").
								contactPerson("akashem").
								productScope(kubernetes).
								enhancementPR("https://github.com/kubernetes/enhancements/issues/4006").
								enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
								mustRegister()

	FeatureGateVolumeAttributesClass = newFeatureGate("VolumeAttributesClass").
						reportProblemsToJiraComponent("Storage / Kubernetes External Components").
						contactPerson("dfajmon").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/3751").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateVolumeGroupSnapshot = newFeatureGate("VolumeGroupSnapshot").
					reportProblemsToJiraComponent("Storage / Kubernetes External Components").
					contactPerson("fbertina").
					productScope(kubernetes).
					enhancementPR("https://github.com/kubernetes/enhancements/issues/3476").
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateExternalOIDC = newFeatureGate("ExternalOIDC").
				reportProblemsToJiraComponent("authentication").
				contactPerson("liouk").
				productScope(ocpSpecific).
				enhancementPR("https://github.com/openshift/enhancements/pull/1596").
				enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
				enableForClusterProfile(Hypershift, configv1.Default, configv1.TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateExample = newFeatureGate("Example").
				reportProblemsToJiraComponent("cluster-config").
				contactPerson("deads").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateExample2 = newFeatureGate("Example2").
				reportProblemsToJiraComponent("cluster-config").
				contactPerson("JoelSpeed").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enableIn(configv1.DevPreviewNoUpgrade).
				mustRegister()

	FeatureGatePlatformOperators = newFeatureGate("PlatformOperators").
					reportProblemsToJiraComponent("olm").
					contactPerson("joe").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNewOLM = newFeatureGate("NewOLM").
				reportProblemsToJiraComponent("olm").
				contactPerson("joe").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enableForClusterProfile(SelfManaged, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade, configv1.Default).
				mustRegister()

	FeatureGateInsightsOnDemandDataGather = newFeatureGate("InsightsOnDemandDataGather").
						reportProblemsToJiraComponent("insights").
						contactPerson("tremes").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateBareMetalLoadBalancer = newFeatureGate("BareMetalLoadBalancer").
						reportProblemsToJiraComponent("metal").
						contactPerson("EmilienM").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateInsightsConfig = newFeatureGate("InsightsConfig").
					reportProblemsToJiraComponent("insights").
					contactPerson("tremes").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNodeDisruptionPolicy = newFeatureGate("NodeDisruptionPolicy").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("jerzhang").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateMetricsCollectionProfiles = newFeatureGate("MetricsCollectionProfiles").
						reportProblemsToJiraComponent("Monitoring").
						contactPerson("rexagod").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateVSphereDriverConfiguration = newFeatureGate("VSphereDriverConfiguration").
						reportProblemsToJiraComponent("Storage / Kubernetes External Components").
						contactPerson("rbednar").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateClusterAPIInstallIBMCloud = newFeatureGate("ClusterAPIInstallIBMCloud").
						reportProblemsToJiraComponent("Installer").
						contactPerson("cjschaef").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						mustRegister()

	FeatureGateChunkSizeMiB = newFeatureGate("ChunkSizeMiB").
				reportProblemsToJiraComponent("Image Registry").
				contactPerson("flavianmissi").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateMachineAPIMigration = newFeatureGate("MachineAPIMigration").
					reportProblemsToJiraComponent("OCPCLOUD").
					contactPerson("jspeed").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					mustRegister()

	FeatureGatePersistentIPsForVirtualization = newFeatureGate("PersistentIPsForVirtualization").
							reportProblemsToJiraComponent("CNV Network").
							contactPerson("mduarted").
							productScope(ocpSpecific).
							enhancementPR(legacyFeatureGateWithoutEnhancement).
							enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateClusterMonitoringConfig = newFeatureGate("ClusterMonitoringConfig").
						reportProblemsToJiraComponent("Monitoring").
						contactPerson("marioferh").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateMultiArchInstallAWS = newFeatureGate("MultiArchInstallAWS").
					reportProblemsToJiraComponent("Installer").
					contactPerson("r4f4").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateMultiArchInstallAzure = newFeatureGate("MultiArchInstallAzure").
						reportProblemsToJiraComponent("Installer").
						contactPerson("r4f4").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						mustRegister()

	FeatureGateMultiArchInstallGCP = newFeatureGate("MultiArchInstallGCP").
					reportProblemsToJiraComponent("Installer").
					contactPerson("r4f4").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateIngressControllerLBSubnetsAWS = newFeatureGate("IngressControllerLBSubnetsAWS").
							reportProblemsToJiraComponent("Routing").
							contactPerson("miciah").
							productScope(ocpSpecific).
							enhancementPR(legacyFeatureGateWithoutEnhancement).
							enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateAWSEFSDriverVolumeMetrics = newFeatureGate("AWSEFSDriverVolumeMetrics").
						reportProblemsToJiraComponent("Storage / Kubernetes External Components").
						contactPerson("fbertina").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.Default, configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateImageStreamImportMode = newFeatureGate("ImageStreamImportMode").
						reportProblemsToJiraComponent("Multi-Arch").
						contactPerson("psundara").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateUserNamespacesSupport = newFeatureGate("UserNamespacesSupport").
						reportProblemsToJiraComponent("Node").
						contactPerson("haircommander").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/127").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateUserNamespacesPodSecurityStandards = newFeatureGate("UserNamespacesPodSecurityStandards").
							reportProblemsToJiraComponent("Node").
							contactPerson("haircommander").
							productScope(kubernetes).
							enhancementPR("https://github.com/kubernetes/enhancements/issues/127").
							enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateProcMountType = newFeatureGate("ProcMountType").
					reportProblemsToJiraComponent("Node").
					contactPerson("haircommander").
					productScope(kubernetes).
					enhancementPR("https://github.com/kubernetes/enhancements/issues/4265").
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateVSphereMultiNetworks = newFeatureGate("VSphereMultiNetworks").
					reportProblemsToJiraComponent("SPLAT").
					contactPerson("rvanderp").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateIngressControllerDynamicConfigurationManager = newFeatureGate("IngressControllerDynamicConfigurationManager").
								reportProblemsToJiraComponent("Networking/router").
								contactPerson("miciah").
								productScope(ocpSpecific).
								enhancementPR(legacyFeatureGateWithoutEnhancement).
								enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
								mustRegister()

	FeatureGateMinimumKubeletVersion = newFeatureGate("MinimumKubeletVersion").
						reportProblemsToJiraComponent("Node").
						contactPerson("haircommander").
						productScope(ocpSpecific).
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						enhancementPR("https://github.com/openshift/enhancements/pull/1697").
						mustRegister()

	FeatureGateNutanixMultiSubnets = newFeatureGate("NutanixMultiSubnets").
					reportProblemsToJiraComponent("Cloud Compute / Nutanix Provider").
					contactPerson("yanhli").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1711").
					enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateKMSEncryptionProvider = newFeatureGate("KMSEncryptionProvider").
						reportProblemsToJiraComponent("kube-apiserver").
						contactPerson("swghosh").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1682").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateHighlyAvailableArbiter = newFeatureGate("HighlyAvailableArbiter").
						reportProblemsToJiraComponent("TwoNode / Arbiter").
						contactPerson("eggfoobar").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1674").
						enableIn(configv1.DevPreviewNoUpgrade, configv1.TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateCVOConfiguration = newFeatureGate("ClusterVersionOperatorConfiguration").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("dhurta").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1492").
					enableIn(configv1.DevPreviewNoUpgrade).
					mustRegister()
)
