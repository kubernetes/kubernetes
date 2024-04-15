package v1

import "fmt"

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

type ClusterProfileName string

var (
	Hypershift         = ClusterProfileName("include.release.openshift.io/ibm-cloud-managed")
	SelfManaged        = ClusterProfileName("include.release.openshift.io/self-managed-high-availability")
	AllClusterProfiles = []ClusterProfileName{Hypershift, SelfManaged}
)

type OwningProduct string

var (
	ocpSpecific = OwningProduct("OCP")
	kubernetes  = OwningProduct("Kubernetes")
)

type featureGateBuilder struct {
	name                string
	owningJiraComponent string
	responsiblePerson   string
	owningProduct       OwningProduct

	statusByClusterProfileByFeatureSet map[ClusterProfileName]map[FeatureSet]bool
}

// newFeatureGate featuregate are disabled in every FeatureSet and selectively enabled
func newFeatureGate(name string) *featureGateBuilder {
	b := &featureGateBuilder{
		name:                               name,
		statusByClusterProfileByFeatureSet: map[ClusterProfileName]map[FeatureSet]bool{},
	}
	for _, clusterProfile := range AllClusterProfiles {
		byFeatureSet := map[FeatureSet]bool{}
		for _, featureSet := range AllFixedFeatureSets {
			byFeatureSet[featureSet] = false
		}
		b.statusByClusterProfileByFeatureSet[clusterProfile] = byFeatureSet
	}
	return b
}

func (b *featureGateBuilder) reportProblemsToJiraComponent(owningJiraComponent string) *featureGateBuilder {
	b.owningJiraComponent = owningJiraComponent
	return b
}

func (b *featureGateBuilder) contactPerson(responsiblePerson string) *featureGateBuilder {
	b.responsiblePerson = responsiblePerson
	return b
}

func (b *featureGateBuilder) productScope(owningProduct OwningProduct) *featureGateBuilder {
	b.owningProduct = owningProduct
	return b
}

func (b *featureGateBuilder) enableIn(featureSets ...FeatureSet) *featureGateBuilder {
	for clusterProfile := range b.statusByClusterProfileByFeatureSet {
		for _, featureSet := range featureSets {
			b.statusByClusterProfileByFeatureSet[clusterProfile][featureSet] = true
		}
	}
	return b
}

func (b *featureGateBuilder) enableForClusterProfile(clusterProfile ClusterProfileName, featureSets ...FeatureSet) *featureGateBuilder {
	for _, featureSet := range featureSets {
		b.statusByClusterProfileByFeatureSet[clusterProfile][featureSet] = true
	}
	return b
}

func (b *featureGateBuilder) register() (FeatureGateName, error) {
	if len(b.name) == 0 {
		return "", fmt.Errorf("missing name")
	}
	if len(b.owningJiraComponent) == 0 {
		return "", fmt.Errorf("missing owningJiraComponent")
	}
	if len(b.responsiblePerson) == 0 {
		return "", fmt.Errorf("missing responsiblePerson")
	}
	if len(b.owningProduct) == 0 {
		return "", fmt.Errorf("missing owningProduct")
	}

	featureGateName := FeatureGateName(b.name)
	description := FeatureGateDescription{
		FeatureGateAttributes: FeatureGateAttributes{
			Name: featureGateName,
		},
		OwningJiraComponent: b.owningJiraComponent,
		ResponsiblePerson:   b.responsiblePerson,
		OwningProduct:       b.owningProduct,
	}

	// statusByClusterProfileByFeatureSet is initialized by constructor to be false for every combination
	for clusterProfile, byFeatureSet := range b.statusByClusterProfileByFeatureSet {
		for featureSet, enabled := range byFeatureSet {
			if _, ok := allFeatureGates[clusterProfile]; !ok {
				allFeatureGates[clusterProfile] = map[FeatureSet]*FeatureGateEnabledDisabled{}
			}
			if _, ok := allFeatureGates[clusterProfile][featureSet]; !ok {
				allFeatureGates[clusterProfile][featureSet] = &FeatureGateEnabledDisabled{}
			}

			if enabled {
				allFeatureGates[clusterProfile][featureSet].Enabled = append(allFeatureGates[clusterProfile][featureSet].Enabled, description)
			} else {
				allFeatureGates[clusterProfile][featureSet].Disabled = append(allFeatureGates[clusterProfile][featureSet].Disabled, description)
			}
		}
	}

	return featureGateName, nil
}

func (b *featureGateBuilder) mustRegister() FeatureGateName {
	ret, err := b.register()
	if err != nil {
		panic(err)
	}
	return ret
}

func FeatureSets(clusterProfile ClusterProfileName, featureSet FeatureSet) (*FeatureGateEnabledDisabled, error) {
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

func AllFeatureSets() map[ClusterProfileName]map[FeatureSet]*FeatureGateEnabledDisabled {
	ret := map[ClusterProfileName]map[FeatureSet]*FeatureGateEnabledDisabled{}

	for clusterProfile, byFeatureSet := range allFeatureGates {
		newByFeatureSet := map[FeatureSet]*FeatureGateEnabledDisabled{}

		for featureSet, enabledDisabled := range byFeatureSet {
			newByFeatureSet[featureSet] = enabledDisabled.DeepCopy()
		}
		ret[clusterProfile] = newByFeatureSet
	}

	return ret
}

var (
	allFeatureGates = map[ClusterProfileName]map[FeatureSet]*FeatureGateEnabledDisabled{}

	FeatureGateServiceAccountTokenNodeBindingValidation = newFeatureGate("ServiceAccountTokenNodeBindingValidation").
								reportProblemsToJiraComponent("apiserver-auth").
								contactPerson("stlaz").
								productScope(kubernetes).
								enableIn(TechPreviewNoUpgrade).
								mustRegister()

	FeatureGateServiceAccountTokenNodeBinding = newFeatureGate("ServiceAccountTokenNodeBinding").
							reportProblemsToJiraComponent("apiserver-auth").
							contactPerson("stlaz").
							productScope(kubernetes).
							enableIn(TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateServiceAccountTokenPodNodeInfo = newFeatureGate("ServiceAccountTokenPodNodeInfo").
							reportProblemsToJiraComponent("apiserver-auth").
							contactPerson("stlaz").
							productScope(kubernetes).
							enableIn(TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateValidatingAdmissionPolicy = newFeatureGate("ValidatingAdmissionPolicy").
						reportProblemsToJiraComponent("kube-apiserver").
						contactPerson("benluddy").
						productScope(kubernetes).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateGatewayAPI = newFeatureGate("GatewayAPI").
				reportProblemsToJiraComponent("Routing").
				contactPerson("miciah").
				productScope(ocpSpecific).
				enableIn(TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateOpenShiftPodSecurityAdmission = newFeatureGate("OpenShiftPodSecurityAdmission").
							reportProblemsToJiraComponent("auth").
							contactPerson("stlaz").
							productScope(ocpSpecific).
							enableIn(Default, TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateExternalCloudProvider = newFeatureGate("ExternalCloudProvider").
						reportProblemsToJiraComponent("cloud-provider").
						contactPerson("jspeed").
						productScope(ocpSpecific).
						enableIn(Default, TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateExternalCloudProviderAzure = newFeatureGate("ExternalCloudProviderAzure").
						reportProblemsToJiraComponent("cloud-provider").
						contactPerson("jspeed").
						productScope(ocpSpecific).
						enableIn(Default, TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateExternalCloudProviderGCP = newFeatureGate("ExternalCloudProviderGCP").
						reportProblemsToJiraComponent("cloud-provider").
						contactPerson("jspeed").
						productScope(ocpSpecific).
						enableIn(Default, TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateExternalCloudProviderExternal = newFeatureGate("ExternalCloudProviderExternal").
							reportProblemsToJiraComponent("cloud-provider").
							contactPerson("elmiko").
							productScope(ocpSpecific).
							enableIn(Default, TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateCSIDriverSharedResource = newFeatureGate("CSIDriverSharedResource").
						reportProblemsToJiraComponent("builds").
						contactPerson("adkaplan").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateBuildCSIVolumes = newFeatureGate("BuildCSIVolumes").
					reportProblemsToJiraComponent("builds").
					contactPerson("adkaplan").
					productScope(ocpSpecific).
					enableIn(Default, TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNodeSwap = newFeatureGate("NodeSwap").
				reportProblemsToJiraComponent("node").
				contactPerson("ehashman").
				productScope(kubernetes).
				enableIn(TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateMachineAPIProviderOpenStack = newFeatureGate("MachineAPIProviderOpenStack").
						reportProblemsToJiraComponent("openstack").
						contactPerson("egarcia").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateInsightsConfigAPI = newFeatureGate("InsightsConfigAPI").
					reportProblemsToJiraComponent("insights").
					contactPerson("tremes").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateDynamicResourceAllocation = newFeatureGate("DynamicResourceAllocation").
						reportProblemsToJiraComponent("scheduling").
						contactPerson("jchaloup").
						productScope(kubernetes).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateAzureWorkloadIdentity = newFeatureGate("AzureWorkloadIdentity").
						reportProblemsToJiraComponent("cloud-credential-operator").
						contactPerson("abutcher").
						productScope(ocpSpecific).
						enableIn(Default, TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateMaxUnavailableStatefulSet = newFeatureGate("MaxUnavailableStatefulSet").
						reportProblemsToJiraComponent("apps").
						contactPerson("atiratree").
						productScope(kubernetes).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateEventedPLEG = newFeatureGate("EventedPLEG").
				reportProblemsToJiraComponent("node").
				contactPerson("sairameshv").
				productScope(kubernetes).
				mustRegister()

	FeatureGatePrivateHostedZoneAWS = newFeatureGate("PrivateHostedZoneAWS").
					reportProblemsToJiraComponent("Routing").
					contactPerson("miciah").
					productScope(ocpSpecific).
					enableIn(Default, TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateSigstoreImageVerification = newFeatureGate("SigstoreImageVerification").
						reportProblemsToJiraComponent("node").
						contactPerson("sgrunert").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateGCPLabelsTags = newFeatureGate("GCPLabelsTags").
					reportProblemsToJiraComponent("Installer").
					contactPerson("bhb").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateAlibabaPlatform = newFeatureGate("AlibabaPlatform").
					reportProblemsToJiraComponent("cloud-provider").
					contactPerson("jspeed").
					productScope(ocpSpecific).
					enableIn(Default, TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateCloudDualStackNodeIPs = newFeatureGate("CloudDualStackNodeIPs").
						reportProblemsToJiraComponent("machine-config-operator/platform-baremetal").
						contactPerson("mkowalsk").
						productScope(kubernetes).
						enableIn(Default, TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateVSphereStaticIPs = newFeatureGate("VSphereStaticIPs").
					reportProblemsToJiraComponent("splat").
					contactPerson("rvanderp3").
					productScope(ocpSpecific).
					enableIn(Default, TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateRouteExternalCertificate = newFeatureGate("RouteExternalCertificate").
						reportProblemsToJiraComponent("router").
						contactPerson("thejasn").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateAdminNetworkPolicy = newFeatureGate("AdminNetworkPolicy").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("tssurya").
					productScope(ocpSpecific).
					enableIn(Default, TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNetworkLiveMigration = newFeatureGate("NetworkLiveMigration").
					reportProblemsToJiraComponent("Networking/ovn-kubernetes").
					contactPerson("pliu").
					productScope(ocpSpecific).
					enableIn(Default, TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNetworkDiagnosticsConfig = newFeatureGate("NetworkDiagnosticsConfig").
						reportProblemsToJiraComponent("Networking/cluster-network-operator").
						contactPerson("kyrtapz").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateHardwareSpeed = newFeatureGate("HardwareSpeed").
					reportProblemsToJiraComponent("etcd").
					contactPerson("hasbro17").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateAutomatedEtcdBackup = newFeatureGate("AutomatedEtcdBackup").
					reportProblemsToJiraComponent("etcd").
					contactPerson("hasbro17").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateMachineAPIOperatorDisableMachineHealthCheckController = newFeatureGate("MachineAPIOperatorDisableMachineHealthCheckController").
										reportProblemsToJiraComponent("ecoproject").
										contactPerson("msluiter").
										productScope(ocpSpecific).
										mustRegister()

	FeatureGateDNSNameResolver = newFeatureGate("DNSNameResolver").
					reportProblemsToJiraComponent("dns").
					contactPerson("miciah").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateVSphereControlPlaneMachineset = newFeatureGate("VSphereControlPlaneMachineSet").
							reportProblemsToJiraComponent("splat").
							contactPerson("rvanderp3").
							productScope(ocpSpecific).
							enableIn(Default, TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateMachineConfigNodes = newFeatureGate("MachineConfigNodes").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("cdoern").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateClusterAPIInstall = newFeatureGate("ClusterAPIInstall").
					reportProblemsToJiraComponent("Installer").
					contactPerson("vincepri").
					productScope(ocpSpecific).
					mustRegister()

	FeatureGateMetricsServer = newFeatureGate("MetricsServer").
					reportProblemsToJiraComponent("Monitoring").
					contactPerson("slashpai").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateInstallAlternateInfrastructureAWS = newFeatureGate("InstallAlternateInfrastructureAWS").
							reportProblemsToJiraComponent("Installer").
							contactPerson("padillon").
							productScope(ocpSpecific).
							enableIn(TechPreviewNoUpgrade).
							mustRegister()

	FeatureGateGCPClusterHostedDNS = newFeatureGate("GCPClusterHostedDNS").
					reportProblemsToJiraComponent("Installer").
					contactPerson("barbacbd").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateMixedCPUsAllocation = newFeatureGate("MixedCPUsAllocation").
					reportProblemsToJiraComponent("NodeTuningOperator").
					contactPerson("titzhak").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateManagedBootImages = newFeatureGate("ManagedBootImages").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("djoshy").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateDisableKubeletCloudCredentialProviders = newFeatureGate("DisableKubeletCloudCredentialProviders").
								reportProblemsToJiraComponent("cloud-provider").
								contactPerson("jspeed").
								productScope(kubernetes).
								enableIn(Default, TechPreviewNoUpgrade).
								mustRegister()

	FeatureGateOnClusterBuild = newFeatureGate("OnClusterBuild").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("dkhater").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateSignatureStores = newFeatureGate("SignatureStores").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("lmohanty").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateKMSv1 = newFeatureGate("KMSv1").
				reportProblemsToJiraComponent("kube-apiserver").
				contactPerson("dgrisonnet").
				productScope(kubernetes).
				enableIn(Default, TechPreviewNoUpgrade).
				mustRegister()

	FeatureGatePinnedImages = newFeatureGate("PinnedImages").
				reportProblemsToJiraComponent("MachineConfigOperator").
				contactPerson("jhernand").
				productScope(ocpSpecific).
				enableIn(TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateUpgradeStatus = newFeatureGate("UpgradeStatus").
					reportProblemsToJiraComponent("Cluster Version Operator").
					contactPerson("pmuller").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateTranslateStreamCloseWebsocketRequests = newFeatureGate("TranslateStreamCloseWebsocketRequests").
								reportProblemsToJiraComponent("kube-apiserver").
								contactPerson("akashem").
								productScope(kubernetes).
								enableIn(TechPreviewNoUpgrade).
								mustRegister()

	FeatureGateVolumeGroupSnapshot = newFeatureGate("VolumeGroupSnapshot").
					reportProblemsToJiraComponent("Storage / Kubernetes External Components").
					contactPerson("fbertina").
					productScope(kubernetes).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateExternalOIDC = newFeatureGate("ExternalOIDC").
				reportProblemsToJiraComponent("authentication").
				contactPerson("stlaz").
				productScope(ocpSpecific).
				enableIn(TechPreviewNoUpgrade).
				enableForClusterProfile(Hypershift, Default, TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateExample = newFeatureGate("Example").
				reportProblemsToJiraComponent("cluster-config").
				contactPerson("deads").
				productScope(ocpSpecific).
				enableIn(TechPreviewNoUpgrade).
				mustRegister()

	FeatureGatePlatformOperators = newFeatureGate("PlatformOperators").
					reportProblemsToJiraComponent("olm").
					contactPerson("joe").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateNewOLM = newFeatureGate("NewOLM").
				reportProblemsToJiraComponent("olm").
				contactPerson("joe").
				productScope(ocpSpecific).
				enableIn(TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateExternalRouteCertificate = newFeatureGate("ExternalRouteCertificate").
						reportProblemsToJiraComponent("network-edge").
						contactPerson("miciah").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateInsightsOnDemandDataGather = newFeatureGate("InsightsOnDemandDataGather").
						reportProblemsToJiraComponent("insights").
						contactPerson("tremes").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateAlertingRules = newFeatureGate("AlertingRules").
					reportProblemsToJiraComponent("Monitoring").
					contactPerson("simon").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateBareMetalLoadBalancer = newFeatureGate("BareMetalLoadBalancer").
						reportProblemsToJiraComponent("metal").
						contactPerson("EmilienM").
						productScope(ocpSpecific).
						enableIn(Default, TechPreviewNoUpgrade).
						mustRegister()

	FeatureGateInsightsConfig = newFeatureGate("InsightsConfig").
					reportProblemsToJiraComponent("insights").
					contactPerson("tremes").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateImagePolicy = newFeatureGate("ImagePolicy").
				reportProblemsToJiraComponent("node").
				contactPerson("rphillips").
				productScope(ocpSpecific).
				enableIn(TechPreviewNoUpgrade).
				mustRegister()

	FeatureGateNodeDisruptionPolicy = newFeatureGate("NodeDisruptionPolicy").
					reportProblemsToJiraComponent("MachineConfigOperator").
					contactPerson("jerzhang").
					productScope(ocpSpecific).
					enableIn(TechPreviewNoUpgrade).
					mustRegister()

	FeatureGateMetricsCollectionProfiles = newFeatureGate("MetricsCollectionProfiles").
						reportProblemsToJiraComponent("Monitoring").
						contactPerson("rexagod").
						productScope(ocpSpecific).
						enableIn(TechPreviewNoUpgrade).
						mustRegister()
)
