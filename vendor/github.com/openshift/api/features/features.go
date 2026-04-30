package features

import (
	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// Generating this many versions future proofs us until at least 2040.
const (
	minOpenshiftVersion uint64 = 4
	maxOpenshiftVersion uint64 = 10
)

func FeatureSets(version uint64, clusterProfile ClusterProfileName, featureSet configv1.FeatureSet) *FeatureGateEnabledDisabled {
	enabledDisabled := &FeatureGateEnabledDisabled{}

	for name, statuses := range allFeatureGates {
		enabled := false

		for _, status := range statuses {
			if status.isEnabled(version, clusterProfile, featureSet) {
				enabled = true
				break
			}
		}

		if enabled {
			enabledDisabled.Enabled = append(enabledDisabled.Enabled, FeatureGateDescription{
				FeatureGateAttributes: configv1.FeatureGateAttributes{
					Name: name,
				},
			})
		} else {
			enabledDisabled.Disabled = append(enabledDisabled.Disabled, FeatureGateDescription{
				FeatureGateAttributes: configv1.FeatureGateAttributes{
					Name: name,
				},
			})
		}
	}

	return enabledDisabled
}

func AllFeatureSets() map[uint64]map[ClusterProfileName]map[configv1.FeatureSet]*FeatureGateEnabledDisabled {
	versions := sets.New[uint64]()
	for version := minOpenshiftVersion; version <= maxOpenshiftVersion; version++ {
		versions.Insert(version)
	}

	clusterProfiles := sets.New[ClusterProfileName](AllClusterProfiles...)
	featureSets := sets.New[configv1.FeatureSet](configv1.AllFixedFeatureSets...)

	// Check for versions explicitly being set among the gates.
	for _, statuses := range allFeatureGates {
		for _, status := range statuses {
			versions.Insert(status.version.UnsortedList()...)
		}
	}

	ret := map[uint64]map[ClusterProfileName]map[configv1.FeatureSet]*FeatureGateEnabledDisabled{}
	for version := range versions {
		ret[version] = map[ClusterProfileName]map[configv1.FeatureSet]*FeatureGateEnabledDisabled{}
		for clusterProfile := range clusterProfiles {
			ret[version][clusterProfile] = map[configv1.FeatureSet]*FeatureGateEnabledDisabled{}
			for featureSet := range featureSets {
				ret[version][clusterProfile][featureSet] = FeatureSets(version, clusterProfile, featureSet)
			}
		}
	}

	return ret
}

var (
	allFeatureGates = map[configv1.FeatureGateName][]featureGateStatus{}

	FeatureGateConsolePluginCSP = newFeatureGate("ConsolePluginContentSecurityPolicy").
					reportProblemsToJiraComponent("Management Console").
					contactPerson("jhadvig").
					productScope(ocpSpecific).
					enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					enhancementPR("https://github.com/openshift/enhancements/pull/1706").
					mustRegister()

	FeatureGateServiceAccountTokenNodeBinding = newFeatureGate("ServiceAccountTokenNodeBinding").
							reportProblemsToJiraComponent("apiserver-auth").
							contactPerson("ibihim").
							productScope(kubernetes).
							enhancementPR("https://github.com/kubernetes/enhancements/issues/4193").
							enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateMutatingAdmissionPolicy = newFeatureGate("MutatingAdmissionPolicy").
						reportProblemsToJiraComponent("kube-apiserver").
						contactPerson("benluddy").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/3962").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateOpenShiftPodSecurityAdmission = newFeatureGate("OpenShiftPodSecurityAdmission").
							reportProblemsToJiraComponent("auth").
							contactPerson("ibihim").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/899").
							enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateBuildCSIVolumes = newFeatureGate("BuildCSIVolumes").
					reportProblemsToJiraComponent("builds").
					contactPerson("adkaplan").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateAzureWorkloadIdentity = newFeatureGate("AzureWorkloadIdentity").
						reportProblemsToJiraComponent("cloud-credential-operator").
						contactPerson("abutcher").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateAzureDedicatedHosts = newFeatureGate("AzureDedicatedHosts").
					reportProblemsToJiraComponent("installer").
					contactPerson("rvanderp3").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1783").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateMaxUnavailableStatefulSet = newFeatureGate("MaxUnavailableStatefulSet").
						reportProblemsToJiraComponent("apps").
						contactPerson("atiratree").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/961").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateEventedPLEG = newFeatureGate("EventedPLEG").
				reportProblemsToJiraComponent("node").
				contactPerson("sairameshv").
				productScope(kubernetes).
				enhancementPR("https://github.com/kubernetes/enhancements/issues/3386").
				mustRegister()

	FeatureGateSigstoreImageVerification = newFeatureGate("SigstoreImageVerification").
						reportProblemsToJiraComponent("node").
						contactPerson("sgrunert").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateSigstoreImageVerificationPKI = newFeatureGate("SigstoreImageVerificationPKI").
						reportProblemsToJiraComponent("node").
						contactPerson("QiWang").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1658").
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateCRIOCredentialProviderConfig = newFeatureGate("CRIOCredentialProviderConfig").
						reportProblemsToJiraComponent("node").
						contactPerson("QiWang").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1861").
						enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
						mustRegister()

	FeatureGateVSphereHostVMGroupZonal = newFeatureGate("VSphereHostVMGroupZonal").
						reportProblemsToJiraComponent("splat").
						contactPerson("jcpowermac").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1677").
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateVSphereMultiDisk = newFeatureGate("VSphereMultiDisk").
					reportProblemsToJiraComponent("splat").
					contactPerson("vr4manta").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1709").
					enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateRouteExternalCertificate = newFeatureGate("RouteExternalCertificate").
						reportProblemsToJiraComponent("router").
						contactPerson("chiragkyal").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateNetworkConnect = newFeatureGate("NetworkConnect").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("tssurya").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/ovn-kubernetes/ovn-kubernetes/pull/5246").
					enable(inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateNoOverlayMode = newFeatureGate("NoOverlayMode").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("pliurh").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1859").
					enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
					mustRegister()

	FeatureGateEVPN = newFeatureGate("EVPN").
			reportProblemsToJiraComponent("Networking/ovn-kubernetes").
			contactPerson("jcaamano").
			productScope(ocpSpecific).
			enhancementPR("https://github.com/openshift/enhancements/pull/1862").
			enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
			mustRegister()

	FeatureGateOVNObservability = newFeatureGate("OVNObservability").
					reportProblemsToJiraComponent("Networking").
					contactPerson("npinaeva").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateBackendQuotaGiB = newFeatureGate("EtcdBackendQuota").
					reportProblemsToJiraComponent("etcd").
					contactPerson("hasbro17").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateAutomatedEtcdBackup = newFeatureGate("AutomatedEtcdBackup").
					reportProblemsToJiraComponent("etcd").
					contactPerson("hasbro17").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
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
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateImageModeStatusReporting = newFeatureGate("ImageModeStatusReporting").
						reportProblemsToJiraComponent("MachineConfigOperator").
						contactPerson("ijanssen").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1809").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateClusterAPIInstall = newFeatureGate("ClusterAPIInstall").
					reportProblemsToJiraComponent("Installer").
					contactPerson("vincepri").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					mustRegister()

	FeatureGateAWSClusterHostedDNS = newFeatureGate("AWSClusterHostedDNS").
					reportProblemsToJiraComponent("Installer").
					contactPerson("barbacbd").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateAzureClusterHostedDNSInstall = newFeatureGate("AzureClusterHostedDNSInstall").
						reportProblemsToJiraComponent("Installer").
						contactPerson("sadasu").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1468").
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateMixedCPUsAllocation = newFeatureGate("MixedCPUsAllocation").
					reportProblemsToJiraComponent("NodeTuningOperator").
					contactPerson("titzhak").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateManagedBootImagesCPMS = newFeatureGate("ManagedBootImagesCPMS").
						reportProblemsToJiraComponent("MachineConfigOperator").
						contactPerson("djoshy").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1818").
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateBootImageSkewEnforcement = newFeatureGate("BootImageSkewEnforcement").
						reportProblemsToJiraComponent("MachineConfigOperator").
						contactPerson("djoshy").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1761").
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateBootcNodeManagement = newFeatureGate("BootcNodeManagement").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("inesqyx").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateSignatureStores = newFeatureGate("SignatureStores").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("lmohanty").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateKMSv1 = newFeatureGate("KMSv1").
				reportProblemsToJiraComponent("kube-apiserver").
				contactPerson("dgrisonnet").
				productScope(kubernetes).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureGateAdditionalStorageConfig = newFeatureGate("AdditionalStorageConfig").
						reportProblemsToJiraComponent("node").
						contactPerson("saschagrunert").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1934").
						enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
						mustRegister()

	FeatureGateUpgradeStatus = newFeatureGate("UpgradeStatus").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("pmuller").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateVolumeGroupSnapshot = newFeatureGate("VolumeGroupSnapshot").
					reportProblemsToJiraComponent("Storage / Kubernetes External Components").
					contactPerson("fbertina").
					productScope(kubernetes).
					enhancementPR("https://github.com/kubernetes/enhancements/issues/3476").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateExternalSnapshotMetadata = newFeatureGate("ExternalSnapshotMetadata").
						reportProblemsToJiraComponent("Storage / Kubernetes External Components").
						contactPerson("jdobson").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/3314").
						enable(inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateExternalOIDC = newFeatureGate("ExternalOIDC").
				reportProblemsToJiraComponent("authentication").
				contactPerson("liouk").
				productScope(ocpSpecific).
				enhancementPR("https://github.com/openshift/enhancements/pull/1596").
				enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureGateExternalOIDCWithAdditionalClaimMappings = newFeatureGate("ExternalOIDCWithUIDAndExtraClaimMappings").
								reportProblemsToJiraComponent("authentication").
								contactPerson("bpalmer").
								productScope(ocpSpecific).
								enhancementPR("https://github.com/openshift/enhancements/pull/1777").
								enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
								mustRegister()

	FeatureGateExternalOIDCWithUpstreamParity = newFeatureGate("ExternalOIDCWithUpstreamParity").
							reportProblemsToJiraComponent("authentication").
							contactPerson("saldawam").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1763").
							enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
							mustRegister()

	FeatureGateExternalOIDCExternalClaimsSourcing = newFeatureGate("ExternalOIDCExternalClaimsSourcing").
							reportProblemsToJiraComponent("authentication").
							contactPerson("bpalmer").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1907").
							enable(inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateExample = newFeatureGate("Example").
				reportProblemsToJiraComponent("cluster-config").
				contactPerson("deads").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureGateExample2 = newFeatureGate("Example2").
				reportProblemsToJiraComponent("cluster-config").
				contactPerson("JoelSpeed").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enable(inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureGateNewOLM = newFeatureGate("NewOLM").
				reportProblemsToJiraComponent("olm").
				contactPerson("joe").
				productScope(ocpSpecific).
				enhancementPR(legacyFeatureGateWithoutEnhancement).
				enable(inClusterProfile(SelfManaged), inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureGateNewOLMCatalogdAPIV1Metas = newFeatureGate("NewOLMCatalogdAPIV1Metas").
						reportProblemsToJiraComponent("olm").
						contactPerson("jordank").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1749").
						enable(inClusterProfile(SelfManaged), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateNewOLMPreflightPermissionChecks = newFeatureGate("NewOLMPreflightPermissionChecks").
							reportProblemsToJiraComponent("olm").
							contactPerson("tshort").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1768").
							enable(inClusterProfile(SelfManaged), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateNewOLMOwnSingleNamespace = newFeatureGate("NewOLMOwnSingleNamespace").
						reportProblemsToJiraComponent("olm").
						contactPerson("nschieder").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1849").
						enable(inClusterProfile(SelfManaged), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateNewOLMWebhookProviderOpenshiftServiceCA = newFeatureGate("NewOLMWebhookProviderOpenshiftServiceCA").
								reportProblemsToJiraComponent("olm").
								contactPerson("pegoncal").
								productScope(ocpSpecific).
								enhancementPR("https://github.com/openshift/enhancements/pull/1844").
								enable(inClusterProfile(SelfManaged), inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
								mustRegister()

	FeatureGateNewOLMBoxCutterRuntime = newFeatureGate("NewOLMBoxCutterRuntime").
						reportProblemsToJiraComponent("olm").
						contactPerson("pegoncal").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1890").
						enable(inClusterProfile(SelfManaged), inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
						mustRegister()

	FeatureGateNewOLMConfigAPI = newFeatureGate("NewOLMConfigAPI").
					reportProblemsToJiraComponent("olm").
					contactPerson("tmshort").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1915").
					enable(inClusterProfile(SelfManaged), inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
					mustRegister()

	FeatureGateInsightsOnDemandDataGather = newFeatureGate("InsightsOnDemandDataGather").
						reportProblemsToJiraComponent("insights").
						contactPerson("tremes").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inDefault(), inOKD(), inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
						mustRegister()

	FeatureGateInsightsConfig = newFeatureGate("InsightsConfig").
					reportProblemsToJiraComponent("insights").
					contactPerson("tremes").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inDefault(), inOKD(), inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
					mustRegister()

	FeatureGateMetricsCollectionProfiles = newFeatureGate("MetricsCollectionProfiles").
						reportProblemsToJiraComponent("Monitoring").
						contactPerson("rexagod").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateClusterAPIInstallIBMCloud = newFeatureGate("ClusterAPIInstallIBMCloud").
						reportProblemsToJiraComponent("Installer").
						contactPerson("cjschaef").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateMachineAPIMigration = newFeatureGate("MachineAPIMigration").
					reportProblemsToJiraComponent("Cloud Compute / Cluster API Providers").
					contactPerson("ddonati").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1465").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateMachineAPIMigrationAWS = newFeatureGate("MachineAPIMigrationAWS").
						reportProblemsToJiraComponent("Cloud Compute / Cluster API Providers").
						contactPerson("ddonati").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1465").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateMachineAPIMigrationOpenStack = newFeatureGate("MachineAPIMigrationOpenStack").
						reportProblemsToJiraComponent("Cloud Compute / OpenStack Provider").
						contactPerson("ddonati").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1465").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()
	FeatureGateMachineAPIMigrationVSphere = newFeatureGate("MachineAPIMigrationVSphere").
						reportProblemsToJiraComponent("SPLAT").
						contactPerson("jcpowermac").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1465").
						enable(inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateClusterAPIMachineManagement = newFeatureGate("ClusterAPIMachineManagement").
						reportProblemsToJiraComponent("Cloud Compute / Cluster API Providers").
						contactPerson("ddonati").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1465").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateClusterAPIMachineManagementAWS = newFeatureGate("ClusterAPIMachineManagementAWS").
							reportProblemsToJiraComponent("Cloud Compute / Cluster API Providers").
							contactPerson("ddonati").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterAPIMachineManagementAzure = newFeatureGate("ClusterAPIMachineManagementAzure").
							reportProblemsToJiraComponent("Cloud Compute / Cluster API Providers").
							contactPerson("ddonati").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterAPIMachineManagementBareMetal = newFeatureGate("ClusterAPIMachineManagementBareMetal").
							reportProblemsToJiraComponent("Cloud Compute / BareMetal Provider").
							contactPerson("ddonati").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterAPIMachineManagementGCP = newFeatureGate("ClusterAPIMachineManagementGCP").
							reportProblemsToJiraComponent("Cloud Compute / Cluster API Providers").
							contactPerson("ddonati").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterAPIMachineManagementPowerVS = newFeatureGate("ClusterAPIMachineManagementPowerVS").
							reportProblemsToJiraComponent("Cloud Compute / IBM Provider").
							contactPerson("ddonati").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterAPIMachineManagementOpenStack = newFeatureGate("ClusterAPIMachineManagementOpenStack").
							reportProblemsToJiraComponent("Cloud Compute / OpenStack Provider").
							contactPerson("ddonati").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterAPIMachineManagementVSphere = newFeatureGate("ClusterAPIMachineManagementVSphere").
							reportProblemsToJiraComponent("SPLAT").
							contactPerson("jcpowermac").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterMonitoringConfig = newFeatureGate("ClusterMonitoringConfig").
						reportProblemsToJiraComponent("Monitoring").
						contactPerson("marioferh").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateMultiArchInstallAzure = newFeatureGate("MultiArchInstallAzure").
						reportProblemsToJiraComponent("Installer").
						contactPerson("r4f4").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						mustRegister()

	FeatureGateImageStreamImportMode = newFeatureGate("ImageStreamImportMode").
						reportProblemsToJiraComponent("Multi-Arch").
						contactPerson("psundara").
						productScope(ocpSpecific).
						enhancementPR(legacyFeatureGateWithoutEnhancement).
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateUserNamespacesSupport = newFeatureGate("UserNamespacesSupport").
						reportProblemsToJiraComponent("Node").
						contactPerson("haircommander").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/127").
						enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	// Note: this feature is perma-alpha, but it is safe and desireable to enable.
	// It was an oversight in upstream to not remove the feature gate after the version skew became safe in 1.33.
	// See https://github.com/kubernetes/enhancements/tree/d4226c42/keps/sig-node/127-user-namespaces#pod-security-standards-pss-integration
	FeatureGateUserNamespacesPodSecurityStandards = newFeatureGate("UserNamespacesPodSecurityStandards").
							reportProblemsToJiraComponent("Node").
							contactPerson("haircommander").
							productScope(kubernetes).
							enhancementPR("https://github.com/kubernetes/enhancements/issues/127").
							enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateVSphereMultiNetworks = newFeatureGate("VSphereMultiNetworks").
					reportProblemsToJiraComponent("SPLAT").
					contactPerson("rvanderp").
					productScope(ocpSpecific).
					enhancementPR(legacyFeatureGateWithoutEnhancement).
					enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateIngressControllerDynamicConfigurationManager = newFeatureGate("IngressControllerDynamicConfigurationManager").
								reportProblemsToJiraComponent("Networking/router").
								contactPerson("miciah").
								productScope(ocpSpecific).
								enhancementPR("https://github.com/openshift/enhancements/pull/1687").
								enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
								mustRegister()

	FeatureGateMinimumKubeletVersion = newFeatureGate("MinimumKubeletVersion").
						reportProblemsToJiraComponent("Node").
						contactPerson("haircommander").
						productScope(ocpSpecific).
						enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
						enhancementPR("https://github.com/openshift/enhancements/pull/1697").
						mustRegister()

	FeatureGateNutanixMultiSubnets = newFeatureGate("NutanixMultiSubnets").
					reportProblemsToJiraComponent("Cloud Compute / Nutanix Provider").
					contactPerson("yanhli").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1711").
					enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
					mustRegister()

	FeatureGateKMSEncryptionProvider = newFeatureGate("KMSEncryptionProvider").
						reportProblemsToJiraComponent("kube-apiserver").
						contactPerson("swghosh").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1682").
						enable(inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateKMSEncryption = newFeatureGate("KMSEncryption").
					reportProblemsToJiraComponent("kube-apiserver").
					contactPerson("ardaguclu").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1900").
					enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
					mustRegister()

	FeatureGateCVOConfiguration = newFeatureGate("ClusterVersionOperatorConfiguration").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("dhurta").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1492").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateClusterUpdateAcceptRisks = newFeatureGate("ClusterUpdateAcceptRisks").
						reportProblemsToJiraComponent("Cluster Version Operator").
						contactPerson("hongkliu").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1807").
						enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
						mustRegister()

	FeatureGateClusterUpdatePreflight = newFeatureGate("ClusterUpdatePreflight").
						reportProblemsToJiraComponent("Cluster Version Operator").
						contactPerson("fao89").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1930").
						enable(inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateGCPCustomAPIEndpoints = newFeatureGate("GCPCustomAPIEndpoints").
						reportProblemsToJiraComponent("Installer").
						contactPerson("barbacbd").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1492").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateDyanmicServiceEndpointIBMCloud = newFeatureGate("DyanmicServiceEndpointIBMCloud").
							reportProblemsToJiraComponent("Cloud Compute / IBM Provider").
							contactPerson("jared-hayes-dev").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1712").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateSELinuxMount = newFeatureGate("SELinuxMount").
				reportProblemsToJiraComponent("Storage / Kubernetes").
				contactPerson("jsafrane").
				productScope(kubernetes).
				enhancementPR("https://github.com/kubernetes/enhancements/issues/1710").
				enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureGateDualReplica = newFeatureGate("DualReplica").
				reportProblemsToJiraComponent("Two Node Fencing").
				contactPerson("jaypoulz").
				productScope(ocpSpecific).
				enhancementPR("https://github.com/openshift/enhancements/pull/1675").
				enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureShortCertRotation = newFeatureGate("ShortCertRotation").
					reportProblemsToJiraComponent("kube-apiserver").
					contactPerson("vrutkovs").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1670").
					mustRegister()

	FeatureGateVSphereConfigurableMaxAllowedBlockVolumesPerNode = newFeatureGate("VSphereConfigurableMaxAllowedBlockVolumesPerNode").
									reportProblemsToJiraComponent("Storage / Kubernetes External Components").
									contactPerson("rbednar").
									productScope(ocpSpecific).
									enhancementPR("https://github.com/openshift/enhancements/pull/1748").
									enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
									mustRegister()

	FeatureGateAzureMultiDisk = newFeatureGate("AzureMultiDisk").
					reportProblemsToJiraComponent("splat").
					contactPerson("jcpowermac").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1779").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateStoragePerformantSecurityPolicy = newFeatureGate("StoragePerformantSecurityPolicy").
							reportProblemsToJiraComponent("Storage / Kubernetes External Components").
							contactPerson("hekumar").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1804").
							enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateMultiDiskSetup = newFeatureGate("MultiDiskSetup").
					reportProblemsToJiraComponent("splat").
					contactPerson("jcpowermac").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1805").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateAWSDedicatedHosts = newFeatureGate("AWSDedicatedHosts").
					reportProblemsToJiraComponent("splat").
					contactPerson("rvanderp3").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1781").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateVSphereMixedNodeEnv = newFeatureGate("VSphereMixedNodeEnv").
					reportProblemsToJiraComponent("splat").
					contactPerson("vr4manta").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1772").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateVSphereMultiVCenterDay2 = newFeatureGate("VSphereMultiVCenterDay2").
						reportProblemsToJiraComponent("splat").
						contactPerson("vr4manta").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1961").
						enable(inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateAWSServiceLBNetworkSecurityGroup = newFeatureGate("AWSServiceLBNetworkSecurityGroup").
							reportProblemsToJiraComponent("Cloud Compute / Cloud Controller Manager").
							contactPerson("mtulio").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1802").
							enable(inClusterProfile(SelfManaged), inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							enable(inClusterProfile(Hypershift), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateNoRegistryClusterInstall = newFeatureGate("NoRegistryClusterInstall").
						reportProblemsToJiraComponent("Installer / Agent based installation").
						contactPerson("andfasano").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1821").
						enable(inClusterProfile(SelfManaged), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateAWSClusterHostedDNSInstall = newFeatureGate("AWSClusterHostedDNSInstall").
						reportProblemsToJiraComponent("Installer").
						contactPerson("barbacbd").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1468").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateGCPCustomAPIEndpointsInstall = newFeatureGate("GCPCustomAPIEndpointsInstall").
						reportProblemsToJiraComponent("Installer").
						contactPerson("barbacbd").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1492").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateIrreconcilableMachineConfig = newFeatureGate("IrreconcilableMachineConfig").
						reportProblemsToJiraComponent("MachineConfigOperator").
						contactPerson("pabrodri").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1785").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()
	FeatureGateAWSDualStackInstall = newFeatureGate("AWSDualStackInstall").
					reportProblemsToJiraComponent("Installer").
					contactPerson("sadasu").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1806").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateAzureDualStackInstall = newFeatureGate("AzureDualStackInstall").
						reportProblemsToJiraComponent("Installer").
						contactPerson("jhixson74").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1806").
						enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateGCPDualStackInstall = newFeatureGate("GCPDualStackInstall").
					reportProblemsToJiraComponent("Installer").
					contactPerson("barbacbd").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1806").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureCBORServingAndStorage = newFeatureGate("CBORServingAndStorage").
					reportProblemsToJiraComponent("kube-apiserver").
					contactPerson("benluddy").
					productScope(kubernetes).
					enhancementPR("https://github.com/kubernetes/enhancements/issues/4222").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureCBORClientsAllowCBOR = newFeatureGate("ClientsAllowCBOR").
					reportProblemsToJiraComponent("kube-apiserver").
					contactPerson("benluddy").
					productScope(kubernetes).
					enhancementPR("https://github.com/kubernetes/enhancements/issues/4222").
					mustRegister()

	FeatureClientsPreferCBOR = newFeatureGate("ClientsPreferCBOR").
					reportProblemsToJiraComponent("kube-apiserver").
					contactPerson("benluddy").
					productScope(kubernetes).
					enhancementPR("https://github.com/kubernetes/enhancements/issues/4222").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureEventTTL = newFeatureGate("EventTTL").
			reportProblemsToJiraComponent("kube-apiserver").
			contactPerson("tjungblu").
			productScope(ocpSpecific).
			enhancementPR("https://github.com/openshift/enhancements/pull/1857").
			enable(inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
			mustRegister()

	FeatureGateMutableCSINodeAllocatableCount = newFeatureGate("MutableCSINodeAllocatableCount").
							reportProblemsToJiraComponent("Storage / Kubernetes External Components").
							contactPerson("jsafrane").
							productScope(kubernetes).
							enhancementPR("https://github.com/kubernetes/enhancements/issues/4876").
							enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade(), inDefault(), inOKD()).
							mustRegister()
	FeatureGateOSStreams = newFeatureGate("OSStreams").
				reportProblemsToJiraComponent("MachineConfigOperator").
				contactPerson("pabrodri").
				productScope(ocpSpecific).
				enhancementPR("https://github.com/openshift/enhancements/pull/1874").
				enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
				mustRegister()

	FeatureGateCRDCompatibilityRequirementOperator = newFeatureGate("CRDCompatibilityRequirementOperator").
							reportProblemsToJiraComponent("Cloud Compute / Cluster API Providers").
							contactPerson("ddonati").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1845").
							enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
							mustRegister()
	FeatureGateOnPremDNSRecords = newFeatureGate("OnPremDNSRecords").
					reportProblemsToJiraComponent("Networking / On-Prem DNS").
					contactPerson("bnemec").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1803").
					enable(inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
					mustRegister()

	FeatureGateProvisioningRequestAvailable = newFeatureGate("ProvisioningRequestAvailable").
						reportProblemsToJiraComponent("Cluster Autoscaler").
						contactPerson("elmiko").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1752").
						enable(inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateHyperShiftOnlyDynamicResourceAllocation = newFeatureGate("HyperShiftOnlyDynamicResourceAllocation").
								reportProblemsToJiraComponent("hypershift").
								contactPerson("csrwng").
								productScope(ocpSpecific).
								enhancementPR("https://github.com/kubernetes/enhancements/issues/4381").
								enable(inClusterProfile(Hypershift), inDefault(), inOKD(), inTechPreviewNoUpgrade(), inDevPreviewNoUpgrade()).
								mustRegister()

	FeatureGateDRAPartitionableDevices = newFeatureGate("DRAPartitionableDevices").
						reportProblemsToJiraComponent("Node").
						contactPerson("harche").
						productScope(kubernetes).
						enhancementPR("https://github.com/kubernetes/enhancements/issues/4815").
						enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
						mustRegister()

	FeatureGateConfigurablePKI = newFeatureGate("ConfigurablePKI").
					reportProblemsToJiraComponent("kube-apiserver").
					contactPerson("sanchezl").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1882").
					enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
					mustRegister()

	FeatureGateClusterAPIControlPlaneInstall = newFeatureGate("ClusterAPIControlPlaneInstall").
							reportProblemsToJiraComponent("Installer / openshift-installer").
							contactPerson("patrickdillon").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1465").
							enable(inDevPreviewNoUpgrade()).
							mustRegister()

	FeatureGateClusterAPIComputeInstall = newFeatureGate("ClusterAPIComputeInstall").
						reportProblemsToJiraComponent("Installer / openshift-installer").
						contactPerson("patrickdillon").
						productScope(ocpSpecific).
						enhancementPR("https://github.com/openshift/enhancements/pull/1465").
						enable(inDevPreviewNoUpgrade()).
						mustRegister()

	FeatureGateAWSEuropeanSovereignCloudInstall = newFeatureGate("AWSEuropeanSovereignCloudInstall").
							reportProblemsToJiraComponent("Installer / openshift-installer").
							contactPerson("tthvo").
							productScope(ocpSpecific).
							enhancementPR("https://github.com/openshift/enhancements/pull/1952").
							enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
							mustRegister()

	FeatureGateGatewayAPIWithoutOLM = newFeatureGate("GatewayAPIWithoutOLM").
					reportProblemsToJiraComponent("Routing").
					contactPerson("miciah").
					productScope(ocpSpecific).
					enhancementPR("https://github.com/openshift/enhancements/pull/1933").
					enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
					mustRegister()

	FeatureGateTLSAdherence = newFeatureGate("TLSAdherence").
				reportProblemsToJiraComponent("HPCASE / TLS Adherence").
				contactPerson("joelanford").
				productScope(ocpSpecific).
				enhancementPR("https://github.com/openshift/enhancements/pull/1910").
				enable(inDevPreviewNoUpgrade(), inTechPreviewNoUpgrade()).
				mustRegister()

        FeatureGateConfidentialCluster = newFeatureGate("ConfidentialCluster").
                                        reportProblemsToJiraComponent("ConfidentialClusters").
                                        contactPerson("fjin").
                                        productScope(ocpSpecific).
                                        enhancementPR("https://github.com/openshift/enhancements/pull/1962").
                                        enable(inDevPreviewNoUpgrade()).
                                        mustRegister()
)
