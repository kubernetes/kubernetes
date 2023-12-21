package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:subresource:status

// Infrastructure holds cluster-wide information about Infrastructure.  The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Infrastructure struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec InfrastructureSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status InfrastructureStatus `json:"status"`
}

// InfrastructureSpec contains settings that apply to the cluster infrastructure.
type InfrastructureSpec struct {
	// cloudConfig is a reference to a ConfigMap containing the cloud provider configuration file.
	// This configuration file is used to configure the Kubernetes cloud provider integration
	// when using the built-in cloud provider integration or the external cloud controller manager.
	// The namespace for this config map is openshift-config.
	//
	// cloudConfig should only be consumed by the kube_cloud_config controller.
	// The controller is responsible for using the user configuration in the spec
	// for various platforms and combining that with the user provided ConfigMap in this field
	// to create a stitched kube cloud config.
	// The controller generates a ConfigMap `kube-cloud-config` in `openshift-config-managed` namespace
	// with the kube cloud config is stored in `cloud.conf` key.
	// All the clients are expected to use the generated ConfigMap only.
	//
	// +optional
	CloudConfig ConfigMapFileReference `json:"cloudConfig"`

	// platformSpec holds desired information specific to the underlying
	// infrastructure provider.
	PlatformSpec PlatformSpec `json:"platformSpec,omitempty"`
}

// InfrastructureStatus describes the infrastructure the cluster is leveraging.
type InfrastructureStatus struct {
	// infrastructureName uniquely identifies a cluster with a human friendly name.
	// Once set it should not be changed. Must be of max length 27 and must have only
	// alphanumeric or hyphen characters.
	InfrastructureName string `json:"infrastructureName"`

	// platform is the underlying infrastructure provider for the cluster.
	//
	// Deprecated: Use platformStatus.type instead.
	Platform PlatformType `json:"platform,omitempty"`

	// platformStatus holds status information specific to the underlying
	// infrastructure provider.
	// +optional
	PlatformStatus *PlatformStatus `json:"platformStatus,omitempty"`

	// etcdDiscoveryDomain is the domain used to fetch the SRV records for discovering
	// etcd servers and clients.
	// For more info: https://github.com/etcd-io/etcd/blob/329be66e8b3f9e2e6af83c123ff89297e49ebd15/Documentation/op-guide/clustering.md#dns-discovery
	// deprecated: as of 4.7, this field is no longer set or honored.  It will be removed in a future release.
	EtcdDiscoveryDomain string `json:"etcdDiscoveryDomain"`

	// apiServerURL is a valid URI with scheme 'https', address and
	// optionally a port (defaulting to 443).  apiServerURL can be used by components like the web console
	// to tell users where to find the Kubernetes API.
	APIServerURL string `json:"apiServerURL"`

	// apiServerInternalURL is a valid URI with scheme 'https',
	// address and optionally a port (defaulting to 443).  apiServerInternalURL can be used by components
	// like kubelets, to contact the Kubernetes API server using the
	// infrastructure provider rather than Kubernetes networking.
	APIServerInternalURL string `json:"apiServerInternalURI"`

	// controlPlaneTopology expresses the expectations for operands that normally run on control nodes.
	// The default is 'HighlyAvailable', which represents the behavior operators have in a "normal" cluster.
	// The 'SingleReplica' mode will be used in single-node deployments
	// and the operators should not configure the operand for highly-available operation
	// The 'External' mode indicates that the control plane is hosted externally to the cluster and that
	// its components are not visible within the cluster.
	// +kubebuilder:default=HighlyAvailable
	// +kubebuilder:validation:Enum=HighlyAvailable;SingleReplica;External
	ControlPlaneTopology TopologyMode `json:"controlPlaneTopology"`

	// infrastructureTopology expresses the expectations for infrastructure services that do not run on control
	// plane nodes, usually indicated by a node selector for a `role` value
	// other than `master`.
	// The default is 'HighlyAvailable', which represents the behavior operators have in a "normal" cluster.
	// The 'SingleReplica' mode will be used in single-node deployments
	// and the operators should not configure the operand for highly-available operation
	// NOTE: External topology mode is not applicable for this field.
	// +kubebuilder:default=HighlyAvailable
	// +kubebuilder:validation:Enum=HighlyAvailable;SingleReplica
	InfrastructureTopology TopologyMode `json:"infrastructureTopology"`

	// cpuPartitioning expresses if CPU partitioning is a currently enabled feature in the cluster.
	// CPU Partitioning means that this cluster can support partitioning workloads to specific CPU Sets.
	// Valid values are "None" and "AllNodes". When omitted, the default value is "None".
	// The default value of "None" indicates that no nodes will be setup with CPU partitioning.
	// The "AllNodes" value indicates that all nodes have been setup with CPU partitioning,
	// and can then be further configured via the PerformanceProfile API.
	// +kubebuilder:default=None
	// +default="None"
	// +kubebuilder:validation:Enum=None;AllNodes
	// +optional
	CPUPartitioning CPUPartitioningMode `json:"cpuPartitioning,omitempty"`
}

// TopologyMode defines the topology mode of the control/infra nodes.
// NOTE: Enum validation is specified in each field that uses this type,
// given that External value is not applicable to the InfrastructureTopology
// field.
type TopologyMode string

const (
	// "HighlyAvailable" is for operators to configure high-availability as much as possible.
	HighlyAvailableTopologyMode TopologyMode = "HighlyAvailable"

	// "SingleReplica" is for operators to avoid spending resources for high-availability purpose.
	SingleReplicaTopologyMode TopologyMode = "SingleReplica"

	// "External" indicates that the component is running externally to the cluster. When specified
	// as the control plane topology, operators should avoid scheduling workloads to masters or assume
	// that any of the control plane components such as kubernetes API server or etcd are visible within
	// the cluster.
	ExternalTopologyMode TopologyMode = "External"
)

// CPUPartitioningMode defines the mode for CPU partitioning
type CPUPartitioningMode string

const (
	// CPUPartitioningNone means that no CPU Partitioning is on in this cluster infrastructure
	CPUPartitioningNone CPUPartitioningMode = "None"

	// CPUPartitioningAllNodes means that all nodes are configured with CPU Partitioning in this cluster
	CPUPartitioningAllNodes CPUPartitioningMode = "AllNodes"
)

// PlatformLoadBalancerType defines the type of load balancer used by the cluster.
type PlatformLoadBalancerType string

const (
	// LoadBalancerTypeUserManaged is a load balancer with control-plane VIPs managed outside of the cluster by the customer.
	LoadBalancerTypeUserManaged PlatformLoadBalancerType = "UserManaged"

	// LoadBalancerTypeOpenShiftManagedDefault is the default load balancer with control-plane VIPs managed by the OpenShift cluster.
	LoadBalancerTypeOpenShiftManagedDefault PlatformLoadBalancerType = "OpenShiftManagedDefault"
)

// PlatformType is a specific supported infrastructure provider.
// +kubebuilder:validation:Enum="";AWS;Azure;BareMetal;GCP;Libvirt;OpenStack;None;VSphere;oVirt;IBMCloud;KubeVirt;EquinixMetal;PowerVS;AlibabaCloud;Nutanix;External
type PlatformType string

const (
	// AWSPlatformType represents Amazon Web Services infrastructure.
	AWSPlatformType PlatformType = "AWS"

	// AzurePlatformType represents Microsoft Azure infrastructure.
	AzurePlatformType PlatformType = "Azure"

	// BareMetalPlatformType represents managed bare metal infrastructure.
	BareMetalPlatformType PlatformType = "BareMetal"

	// GCPPlatformType represents Google Cloud Platform infrastructure.
	GCPPlatformType PlatformType = "GCP"

	// LibvirtPlatformType represents libvirt infrastructure.
	LibvirtPlatformType PlatformType = "Libvirt"

	// OpenStackPlatformType represents OpenStack infrastructure.
	OpenStackPlatformType PlatformType = "OpenStack"

	// NonePlatformType means there is no infrastructure provider.
	NonePlatformType PlatformType = "None"

	// VSpherePlatformType represents VMWare vSphere infrastructure.
	VSpherePlatformType PlatformType = "VSphere"

	// OvirtPlatformType represents oVirt/RHV infrastructure.
	OvirtPlatformType PlatformType = "oVirt"

	// IBMCloudPlatformType represents IBM Cloud infrastructure.
	IBMCloudPlatformType PlatformType = "IBMCloud"

	// KubevirtPlatformType represents KubeVirt/Openshift Virtualization infrastructure.
	KubevirtPlatformType PlatformType = "KubeVirt"

	// EquinixMetalPlatformType represents Equinix Metal infrastructure.
	EquinixMetalPlatformType PlatformType = "EquinixMetal"

	// PowerVSPlatformType represents IBM Power Systems Virtual Servers infrastructure.
	PowerVSPlatformType PlatformType = "PowerVS"

	// AlibabaCloudPlatformType represents Alibaba Cloud infrastructure.
	AlibabaCloudPlatformType PlatformType = "AlibabaCloud"

	// NutanixPlatformType represents Nutanix infrastructure.
	NutanixPlatformType PlatformType = "Nutanix"

	// ExternalPlatformType represents generic infrastructure provider. Platform-specific components should be supplemented separately.
	ExternalPlatformType PlatformType = "External"
)

// IBMCloudProviderType is a specific supported IBM Cloud provider cluster type
type IBMCloudProviderType string

const (
	// Classic  means that the IBM Cloud cluster is using classic infrastructure
	IBMCloudProviderTypeClassic IBMCloudProviderType = "Classic"

	// VPC means that the IBM Cloud cluster is using VPC infrastructure
	IBMCloudProviderTypeVPC IBMCloudProviderType = "VPC"

	// IBMCloudProviderTypeUPI means that the IBM Cloud cluster is using user provided infrastructure.
	// This is utilized in IBM Cloud Satellite environments.
	IBMCloudProviderTypeUPI IBMCloudProviderType = "UPI"
)

// DNSType indicates whether the cluster DNS is hosted by the cluster or Core DNS .
type DNSType string

const (
	// ClusterHosted indicates that a DNS solution other than the default provided by the
	// cloud platform is in use. In this mode, the cluster hosts a DNS solution during installation and the
	// user is expected to provide their own DNS solution post-install.
	// When the DNS solution is `ClusterHosted`, the cluster will continue to use the
	// default Load Balancers provided by the cloud platform.
	ClusterHostedDNSType DNSType = "ClusterHosted"

	// PlatformDefault indicates that the cluster is using the default DNS solution for the
	// cloud platform. OpenShift is responsible for all the LB and DNS configuration needed for the
	// cluster to be functional with no intervention from the user. To accomplish this, OpenShift
	// configures the default LB and DNS solutions provided by the underlying cloud.
	PlatformDefaultDNSType DNSType = "PlatformDefault"
)

// ExternalPlatformSpec holds the desired state for the generic External infrastructure provider.
type ExternalPlatformSpec struct {
	// PlatformName holds the arbitrary string representing the infrastructure provider name, expected to be set at the installation time.
	// This field is solely for informational and reporting purposes and is not expected to be used for decision-making.
	// +kubebuilder:default:="Unknown"
	// +default="Unknown"
	// +kubebuilder:validation:XValidation:rule="oldSelf == 'Unknown' || self == oldSelf",message="platform name cannot be changed once set"
	// +optional
	PlatformName string `json:"platformName,omitempty"`
}

// PlatformSpec holds the desired state specific to the underlying infrastructure provider
// of the current cluster. Since these are used at spec-level for the underlying cluster, it
// is supposed that only one of the spec structs is set.
type PlatformSpec struct {
	// type is the underlying infrastructure provider for the cluster. This
	// value controls whether infrastructure automation such as service load
	// balancers, dynamic volume provisioning, machine creation and deletion, and
	// other integrations are enabled. If None, no infrastructure automation is
	// enabled. Allowed values are "AWS", "Azure", "BareMetal", "GCP", "Libvirt",
	// "OpenStack", "VSphere", "oVirt", "KubeVirt", "EquinixMetal", "PowerVS",
	// "AlibabaCloud", "Nutanix" and "None". Individual components may not support all platforms,
	// and must handle unrecognized platforms as None if they do not support that platform.
	//
	// +unionDiscriminator
	Type PlatformType `json:"type"`

	// AWS contains settings specific to the Amazon Web Services infrastructure provider.
	// +optional
	AWS *AWSPlatformSpec `json:"aws,omitempty"`

	// Azure contains settings specific to the Azure infrastructure provider.
	// +optional
	Azure *AzurePlatformSpec `json:"azure,omitempty"`

	// GCP contains settings specific to the Google Cloud Platform infrastructure provider.
	// +optional
	GCP *GCPPlatformSpec `json:"gcp,omitempty"`

	// BareMetal contains settings specific to the BareMetal platform.
	// +optional
	BareMetal *BareMetalPlatformSpec `json:"baremetal,omitempty"`

	// OpenStack contains settings specific to the OpenStack infrastructure provider.
	// +optional
	OpenStack *OpenStackPlatformSpec `json:"openstack,omitempty"`

	// Ovirt contains settings specific to the oVirt infrastructure provider.
	// +optional
	Ovirt *OvirtPlatformSpec `json:"ovirt,omitempty"`

	// VSphere contains settings specific to the VSphere infrastructure provider.
	// +optional
	VSphere *VSpherePlatformSpec `json:"vsphere,omitempty"`

	// IBMCloud contains settings specific to the IBMCloud infrastructure provider.
	// +optional
	IBMCloud *IBMCloudPlatformSpec `json:"ibmcloud,omitempty"`

	// Kubevirt contains settings specific to the kubevirt infrastructure provider.
	// +optional
	Kubevirt *KubevirtPlatformSpec `json:"kubevirt,omitempty"`

	// EquinixMetal contains settings specific to the Equinix Metal infrastructure provider.
	// +optional
	EquinixMetal *EquinixMetalPlatformSpec `json:"equinixMetal,omitempty"`

	// PowerVS contains settings specific to the IBM Power Systems Virtual Servers infrastructure provider.
	// +optional
	PowerVS *PowerVSPlatformSpec `json:"powervs,omitempty"`

	// AlibabaCloud contains settings specific to the Alibaba Cloud infrastructure provider.
	// +optional
	AlibabaCloud *AlibabaCloudPlatformSpec `json:"alibabaCloud,omitempty"`

	// Nutanix contains settings specific to the Nutanix infrastructure provider.
	// +optional
	Nutanix *NutanixPlatformSpec `json:"nutanix,omitempty"`

	// ExternalPlatformType represents generic infrastructure provider.
	// Platform-specific components should be supplemented separately.
	// +optional
	External *ExternalPlatformSpec `json:"external,omitempty"`
}

// CloudControllerManagerState defines whether Cloud Controller Manager presence is expected or not
type CloudControllerManagerState string

const (
	// Cloud Controller Manager is enabled and expected to be installed.
	// This value indicates that new nodes should be tainted as uninitialized when created,
	// preventing them from running workloads until they are initialized by the cloud controller manager.
	CloudControllerManagerExternal CloudControllerManagerState = "External"

	// Cloud Controller Manager is disabled and not expected to be installed.
	// This value indicates that new nodes should not be tainted
	// and no extra node initialization is expected from the cloud controller manager.
	CloudControllerManagerNone CloudControllerManagerState = "None"
)

// CloudControllerManagerStatus holds the state of Cloud Controller Manager (a.k.a. CCM or CPI) related settings
// +kubebuilder:validation:XValidation:rule="(has(self.state) == has(oldSelf.state)) || (!has(oldSelf.state) && self.state != \"External\")",message="state may not be added or removed once set"
type CloudControllerManagerStatus struct {
	// state determines whether or not an external Cloud Controller Manager is expected to
	// be installed within the cluster.
	// https://kubernetes.io/docs/tasks/administer-cluster/running-cloud-controller/#running-cloud-controller-manager
	//
	// Valid values are "External", "None" and omitted.
	// When set to "External", new nodes will be tainted as uninitialized when created,
	// preventing them from running workloads until they are initialized by the cloud controller manager.
	// When omitted or set to "None", new nodes will be not tainted
	// and no extra initialization from the cloud controller manager is expected.
	// +kubebuilder:validation:Enum="";External;None
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="state is immutable once set"
	// +optional
	State CloudControllerManagerState `json:"state"`
}

// ExternalPlatformStatus holds the current status of the generic External infrastructure provider.
// +kubebuilder:validation:XValidation:rule="has(self.cloudControllerManager) == has(oldSelf.cloudControllerManager)",message="cloudControllerManager may not be added or removed once set"
type ExternalPlatformStatus struct {
	// cloudControllerManager contains settings specific to the external Cloud Controller Manager (a.k.a. CCM or CPI).
	// When omitted, new nodes will be not tainted
	// and no extra initialization from the cloud controller manager is expected.
	// +optional
	CloudControllerManager CloudControllerManagerStatus `json:"cloudControllerManager"`
}

// PlatformStatus holds the current status specific to the underlying infrastructure provider
// of the current cluster. Since these are used at status-level for the underlying cluster, it
// is supposed that only one of the status structs is set.
type PlatformStatus struct {
	// type is the underlying infrastructure provider for the cluster. This
	// value controls whether infrastructure automation such as service load
	// balancers, dynamic volume provisioning, machine creation and deletion, and
	// other integrations are enabled. If None, no infrastructure automation is
	// enabled. Allowed values are "AWS", "Azure", "BareMetal", "GCP", "Libvirt",
	// "OpenStack", "VSphere", "oVirt", "EquinixMetal", "PowerVS", "AlibabaCloud", "Nutanix" and "None".
	// Individual components may not support all platforms, and must handle
	// unrecognized platforms as None if they do not support that platform.
	//
	// This value will be synced with to the `status.platform` and `status.platformStatus.type`.
	// Currently this value cannot be changed once set.
	Type PlatformType `json:"type"`

	// AWS contains settings specific to the Amazon Web Services infrastructure provider.
	// +optional
	AWS *AWSPlatformStatus `json:"aws,omitempty"`

	// Azure contains settings specific to the Azure infrastructure provider.
	// +optional
	Azure *AzurePlatformStatus `json:"azure,omitempty"`

	// GCP contains settings specific to the Google Cloud Platform infrastructure provider.
	// +optional
	GCP *GCPPlatformStatus `json:"gcp,omitempty"`

	// BareMetal contains settings specific to the BareMetal platform.
	// +optional
	BareMetal *BareMetalPlatformStatus `json:"baremetal,omitempty"`

	// OpenStack contains settings specific to the OpenStack infrastructure provider.
	// +optional
	OpenStack *OpenStackPlatformStatus `json:"openstack,omitempty"`

	// Ovirt contains settings specific to the oVirt infrastructure provider.
	// +optional
	Ovirt *OvirtPlatformStatus `json:"ovirt,omitempty"`

	// VSphere contains settings specific to the VSphere infrastructure provider.
	// +optional
	VSphere *VSpherePlatformStatus `json:"vsphere,omitempty"`

	// IBMCloud contains settings specific to the IBMCloud infrastructure provider.
	// +optional
	IBMCloud *IBMCloudPlatformStatus `json:"ibmcloud,omitempty"`

	// Kubevirt contains settings specific to the kubevirt infrastructure provider.
	// +optional
	Kubevirt *KubevirtPlatformStatus `json:"kubevirt,omitempty"`

	// EquinixMetal contains settings specific to the Equinix Metal infrastructure provider.
	// +optional
	EquinixMetal *EquinixMetalPlatformStatus `json:"equinixMetal,omitempty"`

	// PowerVS contains settings specific to the Power Systems Virtual Servers infrastructure provider.
	// +optional
	PowerVS *PowerVSPlatformStatus `json:"powervs,omitempty"`

	// AlibabaCloud contains settings specific to the Alibaba Cloud infrastructure provider.
	// +optional
	AlibabaCloud *AlibabaCloudPlatformStatus `json:"alibabaCloud,omitempty"`

	// Nutanix contains settings specific to the Nutanix infrastructure provider.
	// +optional
	Nutanix *NutanixPlatformStatus `json:"nutanix,omitempty"`

	// External contains settings specific to the generic External infrastructure provider.
	// +optional
	External *ExternalPlatformStatus `json:"external,omitempty"`
}

// AWSServiceEndpoint store the configuration of a custom url to
// override existing defaults of AWS Services.
type AWSServiceEndpoint struct {
	// name is the name of the AWS service.
	// The list of all the service names can be found at https://docs.aws.amazon.com/general/latest/gr/aws-service-information.html
	// This must be provided and cannot be empty.
	//
	// +kubebuilder:validation:Pattern=`^[a-z0-9-]+$`
	Name string `json:"name"`

	// url is fully qualified URI with scheme https, that overrides the default generated
	// endpoint for a client.
	// This must be provided and cannot be empty.
	//
	// +kubebuilder:validation:Pattern=`^https://`
	URL string `json:"url"`
}

// AWSPlatformSpec holds the desired state of the Amazon Web Services infrastructure provider.
// This only includes fields that can be modified in the cluster.
type AWSPlatformSpec struct {
	// serviceEndpoints list contains custom endpoints which will override default
	// service endpoint of AWS Services.
	// There must be only one ServiceEndpoint for a service.
	// +listType=atomic
	// +optional
	ServiceEndpoints []AWSServiceEndpoint `json:"serviceEndpoints,omitempty"`
}

// AWSPlatformStatus holds the current status of the Amazon Web Services infrastructure provider.
type AWSPlatformStatus struct {
	// region holds the default AWS region for new AWS resources created by the cluster.
	Region string `json:"region"`

	// ServiceEndpoints list contains custom endpoints which will override default
	// service endpoint of AWS Services.
	// There must be only one ServiceEndpoint for a service.
	// +listType=atomic
	// +optional
	ServiceEndpoints []AWSServiceEndpoint `json:"serviceEndpoints,omitempty"`

	// resourceTags is a list of additional tags to apply to AWS resources created for the cluster.
	// See https://docs.aws.amazon.com/general/latest/gr/aws_tagging.html for information on tagging AWS resources.
	// AWS supports a maximum of 50 tags per resource. OpenShift reserves 25 tags for its use, leaving 25 tags
	// available for the user.
	// +kubebuilder:validation:MaxItems=25
	// +listType=atomic
	// +optional
	ResourceTags []AWSResourceTag `json:"resourceTags,omitempty"`
}

// AWSResourceTag is a tag to apply to AWS resources created for the cluster.
type AWSResourceTag struct {
	// key is the key of the tag
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	// +kubebuilder:validation:Pattern=`^[0-9A-Za-z_.:/=+-@]+$`
	// +required
	Key string `json:"key"`
	// value is the value of the tag.
	// Some AWS service do not support empty values. Since tags are added to resources in many services, the
	// length of the tag value must meet the requirements of all services.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:Pattern=`^[0-9A-Za-z_.:/=+-@]+$`
	// +required
	Value string `json:"value"`
}

// AzurePlatformSpec holds the desired state of the Azure infrastructure provider.
// This only includes fields that can be modified in the cluster.
type AzurePlatformSpec struct{}

// AzurePlatformStatus holds the current status of the Azure infrastructure provider.
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.resourceTags) && !has(self.resourceTags) || has(oldSelf.resourceTags) && has(self.resourceTags)",message="resourceTags may only be configured during installation"
type AzurePlatformStatus struct {
	// resourceGroupName is the Resource Group for new Azure resources created for the cluster.
	ResourceGroupName string `json:"resourceGroupName"`

	// networkResourceGroupName is the Resource Group for network resources like the Virtual Network and Subnets used by the cluster.
	// If empty, the value is same as ResourceGroupName.
	// +optional
	NetworkResourceGroupName string `json:"networkResourceGroupName,omitempty"`

	// cloudName is the name of the Azure cloud environment which can be used to configure the Azure SDK
	// with the appropriate Azure API endpoints.
	// If empty, the value is equal to `AzurePublicCloud`.
	// +optional
	CloudName AzureCloudEnvironment `json:"cloudName,omitempty"`

	// armEndpoint specifies a URL to use for resource management in non-soverign clouds such as Azure Stack.
	// +optional
	ARMEndpoint string `json:"armEndpoint,omitempty"`

	// resourceTags is a list of additional tags to apply to Azure resources created for the cluster.
	// See https://docs.microsoft.com/en-us/rest/api/resources/tags for information on tagging Azure resources.
	// Due to limitations on Automation, Content Delivery Network, DNS Azure resources, a maximum of 15 tags
	// may be applied. OpenShift reserves 5 tags for internal use, allowing 10 tags for user configuration.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:rule="self.all(x, x in oldSelf) && oldSelf.all(x, x in self)",message="resourceTags are immutable and may only be configured during installation"
	// +listType=atomic
	// +optional
	ResourceTags []AzureResourceTag `json:"resourceTags,omitempty"`
}

// AzureResourceTag is a tag to apply to Azure resources created for the cluster.
type AzureResourceTag struct {
	// key is the key part of the tag. A tag key can have a maximum of 128 characters and cannot be empty. Key
	// must begin with a letter, end with a letter, number or underscore, and must contain only alphanumeric
	// characters and the following special characters `_ . -`.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	// +kubebuilder:validation:Pattern=`^[a-zA-Z]([0-9A-Za-z_.-]*[0-9A-Za-z_])?$`
	Key string `json:"key"`
	// value is the value part of the tag. A tag value can have a maximum of 256 characters and cannot be empty. Value
	// must contain only alphanumeric characters and the following special characters `_ + , - . / : ; < = > ? @`.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:Pattern=`^[0-9A-Za-z_.=+-@]+$`
	Value string `json:"value"`
}

// AzureCloudEnvironment is the name of the Azure cloud environment
// +kubebuilder:validation:Enum="";AzurePublicCloud;AzureUSGovernmentCloud;AzureChinaCloud;AzureGermanCloud;AzureStackCloud
type AzureCloudEnvironment string

const (
	// AzurePublicCloud is the general-purpose, public Azure cloud environment.
	AzurePublicCloud AzureCloudEnvironment = "AzurePublicCloud"

	// AzureUSGovernmentCloud is the Azure cloud environment for the US government.
	AzureUSGovernmentCloud AzureCloudEnvironment = "AzureUSGovernmentCloud"

	// AzureChinaCloud is the Azure cloud environment used in China.
	AzureChinaCloud AzureCloudEnvironment = "AzureChinaCloud"

	// AzureGermanCloud is the Azure cloud environment used in Germany.
	AzureGermanCloud AzureCloudEnvironment = "AzureGermanCloud"

	// AzureStackCloud is the Azure cloud environment used at the edge and on premises.
	AzureStackCloud AzureCloudEnvironment = "AzureStackCloud"
)

// GCPPlatformSpec holds the desired state of the Google Cloud Platform infrastructure provider.
// This only includes fields that can be modified in the cluster.
type GCPPlatformSpec struct{}

// GCPPlatformStatus holds the current status of the Google Cloud Platform infrastructure provider.
// +openshift:validation:FeatureSetAwareXValidation:featureSet=CustomNoUpgrade;TechPreviewNoUpgrade,rule="!has(oldSelf.resourceLabels) && !has(self.resourceLabels) || has(oldSelf.resourceLabels) && has(self.resourceLabels)",message="resourceLabels may only be configured during installation"
// +openshift:validation:FeatureSetAwareXValidation:featureSet=CustomNoUpgrade;TechPreviewNoUpgrade,rule="!has(oldSelf.resourceTags) && !has(self.resourceTags) || has(oldSelf.resourceTags) && has(self.resourceTags)",message="resourceTags may only be configured during installation"
type GCPPlatformStatus struct {
	// resourceGroupName is the Project ID for new GCP resources created for the cluster.
	ProjectID string `json:"projectID"`

	// region holds the region for new GCP resources created for the cluster.
	Region string `json:"region"`

	// resourceLabels is a list of additional labels to apply to GCP resources created for the cluster.
	// See https://cloud.google.com/compute/docs/labeling-resources for information on labeling GCP resources.
	// GCP supports a maximum of 64 labels per resource. OpenShift reserves 32 labels for internal use,
	// allowing 32 labels for user configuration.
	// +kubebuilder:validation:MaxItems=32
	// +kubebuilder:validation:XValidation:rule="self.all(x, x in oldSelf) && oldSelf.all(x, x in self)",message="resourceLabels are immutable and may only be configured during installation"
	// +listType=map
	// +listMapKey=key
	// +optional
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	ResourceLabels []GCPResourceLabel `json:"resourceLabels,omitempty"`

	// resourceTags is a list of additional tags to apply to GCP resources created for the cluster.
	// See https://cloud.google.com/resource-manager/docs/tags/tags-overview for information on
	// tagging GCP resources. GCP supports a maximum of 50 tags per resource.
	// +kubebuilder:validation:MaxItems=50
	// +kubebuilder:validation:XValidation:rule="self.all(x, x in oldSelf) && oldSelf.all(x, x in self)",message="resourceTags are immutable and may only be configured during installation"
	// +listType=map
	// +listMapKey=key
	// +optional
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	ResourceTags []GCPResourceTag `json:"resourceTags,omitempty"`

	// This field was introduced and removed under tech preview.
	// To avoid conflicts with serialisation, this field name may never be used again.
	// Tombstone the field as a reminder.
	// ClusterHostedDNS ClusterHostedDNS `json:"clusterHostedDNS,omitempty"`

	// cloudLoadBalancerConfig is a union that contains the IP addresses of API,
	// API-Int and Ingress Load Balancers created on the cloud platform. These
	// values would not be populated on on-prem platforms. These Load Balancer
	// IPs are used to configure the in-cluster DNS instances for API, API-Int
	// and Ingress services. `dnsType` is expected to be set to `ClusterHosted`
	// when these Load Balancer IP addresses are populated and used.
	//
	// +default={"dnsType": "PlatformDefault"}
	// +kubebuilder:default={"dnsType": "PlatformDefault"}
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +optional
	// +nullable
	CloudLoadBalancerConfig *CloudLoadBalancerConfig `json:"cloudLoadBalancerConfig,omitempty"`
}

// GCPResourceLabel is a label to apply to GCP resources created for the cluster.
type GCPResourceLabel struct {
	// key is the key part of the label. A label key can have a maximum of 63 characters and cannot be empty.
	// Label key must begin with a lowercase letter, and must contain only lowercase letters, numeric characters,
	// and the following special characters `_-`. Label key must not have the reserved prefixes `kubernetes-io`
	// and `openshift-io`.
	// +kubebuilder:validation:XValidation:rule="!self.startsWith('openshift-io') && !self.startsWith('kubernetes-io')",message="label keys must not start with either `openshift-io` or `kubernetes-io`"
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[a-z][0-9a-z_-]{0,62}$`
	Key string `json:"key"`

	// value is the value part of the label. A label value can have a maximum of 63 characters and cannot be empty.
	// Value must contain only lowercase letters, numeric characters, and the following special characters `_-`.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[0-9a-z_-]{1,63}$`
	Value string `json:"value"`
}

// GCPResourceTag is a tag to apply to GCP resources created for the cluster.
type GCPResourceTag struct {
	// parentID is the ID of the hierarchical resource where the tags are defined,
	// e.g. at the Organization or the Project level. To find the Organization or Project ID refer to the following pages:
	// https://cloud.google.com/resource-manager/docs/creating-managing-organization#retrieving_your_organization_id,
	// https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects.
	// An OrganizationID must consist of decimal numbers, and cannot have leading zeroes.
	// A ProjectID must be 6 to 30 characters in length, can only contain lowercase letters, numbers,
	// and hyphens, and must start with a letter, and cannot end with a hyphen.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=32
	// +kubebuilder:validation:Pattern=`(^[1-9][0-9]{0,31}$)|(^[a-z][a-z0-9-]{4,28}[a-z0-9]$)`
	ParentID string `json:"parentID"`

	// key is the key part of the tag. A tag key can have a maximum of 63 characters and cannot be empty.
	// Tag key must begin and end with an alphanumeric character, and must contain only uppercase, lowercase
	// alphanumeric characters, and the following special characters `._-`.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[a-zA-Z0-9]([0-9A-Za-z_.-]{0,61}[a-zA-Z0-9])?$`
	Key string `json:"key"`

	// value is the value part of the tag. A tag value can have a maximum of 63 characters and cannot be empty.
	// Tag value must begin and end with an alphanumeric character, and must contain only uppercase, lowercase
	// alphanumeric characters, and the following special characters `_-.@%=+:,*#&(){}[]` and spaces.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[a-zA-Z0-9]([0-9A-Za-z_.@%=+:,*#&()\[\]{}\-\s]{0,61}[a-zA-Z0-9])?$`
	Value string `json:"value"`
}

// CloudLoadBalancerConfig contains an union discriminator indicating the type of DNS
// solution in use within the cluster. When the DNSType is `ClusterHosted`, the cloud's
// Load Balancer configuration needs to be provided so that the DNS solution hosted
// within the cluster can be configured with those values.
// +kubebuilder:validation:XValidation:rule="has(self.dnsType) && self.dnsType != 'ClusterHosted' ? !has(self.clusterHosted) : true",message="clusterHosted is permitted only when dnsType is ClusterHosted"
// +union
type CloudLoadBalancerConfig struct {
	// dnsType indicates the type of DNS solution in use within the cluster. Its default value of
	// `PlatformDefault` indicates that the cluster's DNS is the default provided by the cloud platform.
	// It can be set to `ClusterHosted` to bypass the configuration of the cloud default DNS. In this mode,
	// the cluster needs to provide a self-hosted DNS solution for the cluster's installation to succeed.
	// The cluster's use of the cloud's Load Balancers is unaffected by this setting.
	// The value is immutable after it has been set at install time.
	// Currently, there is no way for the customer to add additional DNS entries into the cluster hosted DNS.
	// Enabling this functionality allows the user to start their own DNS solution outside the cluster after
	// installation is complete. The customer would be responsible for configuring this custom DNS solution,
	// and it can be run in addition to the in-cluster DNS solution.
	// +default="PlatformDefault"
	// +kubebuilder:default:="PlatformDefault"
	// +kubebuilder:validation:Enum="ClusterHosted";"PlatformDefault"
	// +kubebuilder:validation:XValidation:rule="oldSelf == '' || self == oldSelf",message="dnsType is immutable"
	// +optional
	// +unionDiscriminator
	DNSType DNSType `json:"dnsType,omitempty"`

	// clusterHosted holds the IP addresses of API, API-Int and Ingress Load
	// Balancers on Cloud Platforms. The DNS solution hosted within the cluster
	// use these IP addresses to provide resolution for API, API-Int and Ingress
	// services.
	// +optional
	// +unionMember,optional
	ClusterHosted *CloudLoadBalancerIPs `json:"clusterHosted,omitempty"`
}

// CloudLoadBalancerIPs contains the Load Balancer IPs for the cloud's API,
// API-Int and Ingress Load balancers. They will be populated as soon as the
// respective Load Balancers have been configured. These values are utilized
// to configure the DNS solution hosted within the cluster.
type CloudLoadBalancerIPs struct {
	// apiIntLoadBalancerIPs holds Load Balancer IPs for the internal API service.
	// These Load Balancer IP addresses can be IPv4 and/or IPv6 addresses.
	// Entries in the apiIntLoadBalancerIPs must be unique.
	// A maximum of 16 IP addresses are permitted.
	// +kubebuilder:validation:Format=ip
	// +listType=set
	// +kubebuilder:validation:MaxItems=16
	// +optional
	APIIntLoadBalancerIPs []IP `json:"apiIntLoadBalancerIPs,omitempty"`

	// apiLoadBalancerIPs holds Load Balancer IPs for the API service.
	// These Load Balancer IP addresses can be IPv4 and/or IPv6 addresses.
	// Could be empty for private clusters.
	// Entries in the apiLoadBalancerIPs must be unique.
	// A maximum of 16 IP addresses are permitted.
	// +kubebuilder:validation:Format=ip
	// +listType=set
	// +kubebuilder:validation:MaxItems=16
	// +optional
	APILoadBalancerIPs []IP `json:"apiLoadBalancerIPs,omitempty"`

	// ingressLoadBalancerIPs holds IPs for Ingress Load Balancers.
	// These Load Balancer IP addresses can be IPv4 and/or IPv6 addresses.
	// Entries in the ingressLoadBalancerIPs must be unique.
	// A maximum of 16 IP addresses are permitted.
	// +kubebuilder:validation:Format=ip
	// +listType=set
	// +kubebuilder:validation:MaxItems=16
	// +optional
	IngressLoadBalancerIPs []IP `json:"ingressLoadBalancerIPs,omitempty"`
}

// BareMetalPlatformLoadBalancer defines the load balancer used by the cluster on BareMetal platform.
// +union
type BareMetalPlatformLoadBalancer struct {
	// type defines the type of load balancer used by the cluster on BareMetal platform
	// which can be a user-managed or openshift-managed load balancer
	// that is to be used for the OpenShift API and Ingress endpoints.
	// When set to OpenShiftManagedDefault the static pods in charge of API and Ingress traffic load-balancing
	// defined in the machine config operator will be deployed.
	// When set to UserManaged these static pods will not be deployed and it is expected that
	// the load balancer is configured out of band by the deployer.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default.
	// The default value is OpenShiftManagedDefault.
	// +default="OpenShiftManagedDefault"
	// +kubebuilder:default:="OpenShiftManagedDefault"
	// +kubebuilder:validation:Enum:="OpenShiftManagedDefault";"UserManaged"
	// +kubebuilder:validation:XValidation:rule="oldSelf == '' || self == oldSelf",message="type is immutable once set"
	// +optional
	// +unionDiscriminator
	Type PlatformLoadBalancerType `json:"type,omitempty"`
}

// BareMetalPlatformSpec holds the desired state of the BareMetal infrastructure provider.
// This only includes fields that can be modified in the cluster.
type BareMetalPlatformSpec struct{}

// BareMetalPlatformStatus holds the current status of the BareMetal infrastructure provider.
// For more information about the network architecture used with the BareMetal platform type, see:
// https://github.com/openshift/installer/blob/master/docs/design/baremetal/networking-infrastructure.md
type BareMetalPlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	//
	// Deprecated: Use APIServerInternalIPs instead.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// apiServerInternalIPs are the IP addresses to contact the Kubernetes API
	// server that can be used by components inside the cluster, like kubelets
	// using the infrastructure rather than Kubernetes networking. These are the
	// IPs for a self-hosted load balancer in front of the API servers. In dual
	// stack clusters this list contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	APIServerInternalIPs []string `json:"apiServerInternalIPs"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	//
	// Deprecated: Use IngressIPs instead.
	IngressIP string `json:"ingressIP,omitempty"`

	// ingressIPs are the external IPs which route to the default ingress
	// controller. The IPs are suitable targets of a wildcard DNS record used to
	// resolve default route host names. In dual stack clusters this list
	// contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	IngressIPs []string `json:"ingressIPs"`

	// nodeDNSIP is the IP address for the internal DNS used by the
	// nodes. Unlike the one managed by the DNS operator, `NodeDNSIP`
	// provides name resolution for the nodes themselves. There is no DNS-as-a-service for
	// BareMetal deployments. In order to minimize necessary changes to the
	// datacenter DNS, a DNS service is hosted as a static pod to serve those hostnames
	// to the nodes in the cluster.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`

	// loadBalancer defines how the load balancer used by the cluster is configured.
	// +default={"type": "OpenShiftManagedDefault"}
	// +kubebuilder:default={"type": "OpenShiftManagedDefault"}
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +optional
	LoadBalancer *BareMetalPlatformLoadBalancer `json:"loadBalancer,omitempty"`
}

// OpenStackPlatformLoadBalancer defines the load balancer used by the cluster on OpenStack platform.
// +union
type OpenStackPlatformLoadBalancer struct {
	// type defines the type of load balancer used by the cluster on OpenStack platform
	// which can be a user-managed or openshift-managed load balancer
	// that is to be used for the OpenShift API and Ingress endpoints.
	// When set to OpenShiftManagedDefault the static pods in charge of API and Ingress traffic load-balancing
	// defined in the machine config operator will be deployed.
	// When set to UserManaged these static pods will not be deployed and it is expected that
	// the load balancer is configured out of band by the deployer.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default.
	// The default value is OpenShiftManagedDefault.
	// +default="OpenShiftManagedDefault"
	// +kubebuilder:default:="OpenShiftManagedDefault"
	// +kubebuilder:validation:Enum:="OpenShiftManagedDefault";"UserManaged"
	// +kubebuilder:validation:XValidation:rule="oldSelf == '' || self == oldSelf",message="type is immutable once set"
	// +optional
	// +unionDiscriminator
	Type PlatformLoadBalancerType `json:"type,omitempty"`
}

// OpenStackPlatformSpec holds the desired state of the OpenStack infrastructure provider.
// This only includes fields that can be modified in the cluster.
type OpenStackPlatformSpec struct{}

// OpenStackPlatformStatus holds the current status of the OpenStack infrastructure provider.
type OpenStackPlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	//
	// Deprecated: Use APIServerInternalIPs instead.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// apiServerInternalIPs are the IP addresses to contact the Kubernetes API
	// server that can be used by components inside the cluster, like kubelets
	// using the infrastructure rather than Kubernetes networking. These are the
	// IPs for a self-hosted load balancer in front of the API servers. In dual
	// stack clusters this list contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	APIServerInternalIPs []string `json:"apiServerInternalIPs"`

	// cloudName is the name of the desired OpenStack cloud in the
	// client configuration file (`clouds.yaml`).
	CloudName string `json:"cloudName,omitempty"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	//
	// Deprecated: Use IngressIPs instead.
	IngressIP string `json:"ingressIP,omitempty"`

	// ingressIPs are the external IPs which route to the default ingress
	// controller. The IPs are suitable targets of a wildcard DNS record used to
	// resolve default route host names. In dual stack clusters this list
	// contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	IngressIPs []string `json:"ingressIPs"`

	// nodeDNSIP is the IP address for the internal DNS used by the
	// nodes. Unlike the one managed by the DNS operator, `NodeDNSIP`
	// provides name resolution for the nodes themselves. There is no DNS-as-a-service for
	// OpenStack deployments. In order to minimize necessary changes to the
	// datacenter DNS, a DNS service is hosted as a static pod to serve those hostnames
	// to the nodes in the cluster.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`

	// loadBalancer defines how the load balancer used by the cluster is configured.
	// +default={"type": "OpenShiftManagedDefault"}
	// +kubebuilder:default={"type": "OpenShiftManagedDefault"}
	// +optional
	LoadBalancer *OpenStackPlatformLoadBalancer `json:"loadBalancer,omitempty"`
}

// OvirtPlatformLoadBalancer defines the load balancer used by the cluster on Ovirt platform.
// +union
type OvirtPlatformLoadBalancer struct {
	// type defines the type of load balancer used by the cluster on Ovirt platform
	// which can be a user-managed or openshift-managed load balancer
	// that is to be used for the OpenShift API and Ingress endpoints.
	// When set to OpenShiftManagedDefault the static pods in charge of API and Ingress traffic load-balancing
	// defined in the machine config operator will be deployed.
	// When set to UserManaged these static pods will not be deployed and it is expected that
	// the load balancer is configured out of band by the deployer.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default.
	// The default value is OpenShiftManagedDefault.
	// +default="OpenShiftManagedDefault"
	// +kubebuilder:default:="OpenShiftManagedDefault"
	// +kubebuilder:validation:Enum:="OpenShiftManagedDefault";"UserManaged"
	// +kubebuilder:validation:XValidation:rule="oldSelf == '' || self == oldSelf",message="type is immutable once set"
	// +optional
	// +unionDiscriminator
	Type PlatformLoadBalancerType `json:"type,omitempty"`
}

// OvirtPlatformSpec holds the desired state of the oVirt infrastructure provider.
// This only includes fields that can be modified in the cluster.
type OvirtPlatformSpec struct{}

// OvirtPlatformStatus holds the current status of the  oVirt infrastructure provider.
type OvirtPlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	//
	// Deprecated: Use APIServerInternalIPs instead.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// apiServerInternalIPs are the IP addresses to contact the Kubernetes API
	// server that can be used by components inside the cluster, like kubelets
	// using the infrastructure rather than Kubernetes networking. These are the
	// IPs for a self-hosted load balancer in front of the API servers. In dual
	// stack clusters this list contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	APIServerInternalIPs []string `json:"apiServerInternalIPs"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	//
	// Deprecated: Use IngressIPs instead.
	IngressIP string `json:"ingressIP,omitempty"`

	// ingressIPs are the external IPs which route to the default ingress
	// controller. The IPs are suitable targets of a wildcard DNS record used to
	// resolve default route host names. In dual stack clusters this list
	// contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	IngressIPs []string `json:"ingressIPs"`

	// deprecated: as of 4.6, this field is no longer set or honored.  It will be removed in a future release.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`

	// loadBalancer defines how the load balancer used by the cluster is configured.
	// +default={"type": "OpenShiftManagedDefault"}
	// +kubebuilder:default={"type": "OpenShiftManagedDefault"}
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +optional
	LoadBalancer *OvirtPlatformLoadBalancer `json:"loadBalancer,omitempty"`
}

// VSpherePlatformLoadBalancer defines the load balancer used by the cluster on VSphere platform.
// +union
type VSpherePlatformLoadBalancer struct {
	// type defines the type of load balancer used by the cluster on VSphere platform
	// which can be a user-managed or openshift-managed load balancer
	// that is to be used for the OpenShift API and Ingress endpoints.
	// When set to OpenShiftManagedDefault the static pods in charge of API and Ingress traffic load-balancing
	// defined in the machine config operator will be deployed.
	// When set to UserManaged these static pods will not be deployed and it is expected that
	// the load balancer is configured out of band by the deployer.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default.
	// The default value is OpenShiftManagedDefault.
	// +default="OpenShiftManagedDefault"
	// +kubebuilder:default:="OpenShiftManagedDefault"
	// +kubebuilder:validation:Enum:="OpenShiftManagedDefault";"UserManaged"
	// +kubebuilder:validation:XValidation:rule="oldSelf == '' || self == oldSelf",message="type is immutable once set"
	// +optional
	// +unionDiscriminator
	Type PlatformLoadBalancerType `json:"type,omitempty"`
}

// VSpherePlatformFailureDomainSpec holds the region and zone failure domain and
// the vCenter topology of that failure domain.
type VSpherePlatformFailureDomainSpec struct {
	// name defines the arbitrary but unique name
	// of a failure domain.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	Name string `json:"name"`

	// region defines the name of a region tag that will
	// be attached to a vCenter datacenter. The tag
	// category in vCenter must be named openshift-region.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=80
	// +kubebuilder:validation:Required
	Region string `json:"region"`

	// zone defines the name of a zone tag that will
	// be attached to a vCenter cluster. The tag
	// category in vCenter must be named openshift-zone.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=80
	// +kubebuilder:validation:Required
	Zone string `json:"zone"`

	// server is the fully-qualified domain name or the IP address of the vCenter server.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=255
	// ---
	// + Validation is applied via a patch, we validate the format as either ipv4, ipv6 or hostname
	Server string `json:"server"`

	// Topology describes a given failure domain using vSphere constructs
	// +kubebuilder:validation:Required
	Topology VSpherePlatformTopology `json:"topology"`
}

// VSpherePlatformTopology holds the required and optional vCenter objects - datacenter,
// computeCluster, networks, datastore and resourcePool - to provision virtual machines.
type VSpherePlatformTopology struct {
	// datacenter is the name of vCenter datacenter in which virtual machines will be located.
	// The maximum length of the datacenter name is 80 characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=80
	Datacenter string `json:"datacenter"`

	// computeCluster the absolute path of the vCenter cluster
	// in which virtual machine will be located.
	// The absolute path is of the form /<datacenter>/host/<cluster>.
	// The maximum length of the path is 2048 characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:Pattern=`^/.*?/host/.*?`
	ComputeCluster string `json:"computeCluster"`

	// networks is the list of port group network names within this failure domain.
	// Currently, we only support a single interface per RHCOS virtual machine.
	// The available networks (port groups) can be listed using
	// `govc ls 'network/*'`
	// The single interface should be the absolute path of the form
	// /<datacenter>/network/<portgroup>.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxItems=1
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	Networks []string `json:"networks"`

	// datastore is the absolute path of the datastore in which the
	// virtual machine is located.
	// The absolute path is of the form /<datacenter>/datastore/<datastore>
	// The maximum length of the path is 2048 characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:Pattern=`^/.*?/datastore/.*?`
	Datastore string `json:"datastore"`

	// resourcePool is the absolute path of the resource pool where virtual machines will be
	// created. The absolute path is of the form /<datacenter>/host/<cluster>/Resources/<resourcepool>.
	// The maximum length of the path is 2048 characters.
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:Pattern=`^/.*?/host/.*?/Resources.*`
	// +optional
	ResourcePool string `json:"resourcePool,omitempty"`

	// folder is the absolute path of the folder where
	// virtual machines are located. The absolute path
	// is of the form /<datacenter>/vm/<folder>.
	// The maximum length of the path is 2048 characters.
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:Pattern=`^/.*?/vm/.*?`
	// +optional
	Folder string `json:"folder,omitempty"`

	// template is the full inventory path of the virtual machine or template
	// that will be cloned when creating new machines in this failure domain.
	// The maximum length of the path is 2048 characters.
	//
	// When omitted, the template will be calculated by the control plane
	// machineset operator based on the region and zone defined in
	// VSpherePlatformFailureDomainSpec.
	// For example, for zone=zonea, region=region1, and infrastructure name=test,
	// the template path would be calculated as /<datacenter>/vm/test-rhcos-region1-zonea.
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:Pattern=`^/.*?/vm/.*?`
	// +optional
	Template string `json:"template,omitempty"`
}

// VSpherePlatformVCenterSpec stores the vCenter connection fields.
// This is used by the vSphere CCM.
type VSpherePlatformVCenterSpec struct {

	// server is the fully-qualified domain name or the IP address of the vCenter server.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=255
	// ---
	// + Validation is applied via a patch, we validate the format as either ipv4, ipv6 or hostname
	Server string `json:"server"`

	// port is the TCP port that will be used to communicate to
	// the vCenter endpoint.
	// When omitted, this means the user has no opinion and
	// it is up to the platform to choose a sensible default,
	// which is subject to change over time.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=32767
	// +optional
	Port int32 `json:"port,omitempty"`

	// The vCenter Datacenters in which the RHCOS
	// vm guests are located. This field will
	// be used by the Cloud Controller Manager.
	// Each datacenter listed here should be used within
	// a topology.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +listType=set
	Datacenters []string `json:"datacenters"`
}

// VSpherePlatformNodeNetworkingSpec holds the network CIDR(s) and port group name for
// including and excluding IP ranges in the cloud provider.
// This would be used for example when multiple network adapters are attached to
// a guest to help determine which IP address the cloud config manager should use
// for the external and internal node networking.
type VSpherePlatformNodeNetworkingSpec struct {
	// networkSubnetCidr IP address on VirtualMachine's network interfaces included in the fields' CIDRs
	// that will be used in respective status.addresses fields.
	// ---
	// + Validation is applied via a patch, we validate the format as cidr
	// +listType=set
	// +optional
	NetworkSubnetCIDR []string `json:"networkSubnetCidr,omitempty"`

	// network VirtualMachine's VM Network names that will be used to when searching
	// for status.addresses fields. Note that if internal.networkSubnetCIDR and
	// external.networkSubnetCIDR are not set, then the vNIC associated to this network must
	// only have a single IP address assigned to it.
	// The available networks (port groups) can be listed using
	// `govc ls 'network/*'`
	// +optional
	Network string `json:"network,omitempty"`

	// excludeNetworkSubnetCidr IP addresses in subnet ranges will be excluded when selecting
	// the IP address from the VirtualMachine's VM for use in the status.addresses fields.
	// ---
	// + Validation is applied via a patch, we validate the format as cidr
	// +listType=atomic
	// +optional
	ExcludeNetworkSubnetCIDR []string `json:"excludeNetworkSubnetCidr,omitempty"`
}

// VSpherePlatformNodeNetworking holds the external and internal node networking spec.
type VSpherePlatformNodeNetworking struct {
	// external represents the network configuration of the node that is externally routable.
	// +optional
	External VSpherePlatformNodeNetworkingSpec `json:"external"`
	// internal represents the network configuration of the node that is routable only within the cluster.
	// +optional
	Internal VSpherePlatformNodeNetworkingSpec `json:"internal"`
}

// VSpherePlatformSpec holds the desired state of the vSphere infrastructure provider.
// In the future the cloud provider operator, storage operator and machine operator will
// use these fields for configuration.
type VSpherePlatformSpec struct {
	// vcenters holds the connection details for services to communicate with vCenter.
	// Currently, only a single vCenter is supported.
	// ---
	// + If VCenters is not defined use the existing cloud-config configmap defined
	// + in openshift-config.
	// +kubebuilder:validation:MaxItems=1
	// +kubebuilder:validation:MinItems=0
	// +listType=atomic
	// +optional
	VCenters []VSpherePlatformVCenterSpec `json:"vcenters,omitempty"`

	// failureDomains contains the definition of region, zone and the vCenter topology.
	// If this is omitted failure domains (regions and zones) will not be used.
	// +listType=map
	// +listMapKey=name
	// +optional
	FailureDomains []VSpherePlatformFailureDomainSpec `json:"failureDomains,omitempty"`

	// nodeNetworking contains the definition of internal and external network constraints for
	// assigning the node's networking.
	// If this field is omitted, networking defaults to the legacy
	// address selection behavior which is to only support a single address and
	// return the first one found.
	// +optional
	NodeNetworking VSpherePlatformNodeNetworking `json:"nodeNetworking,omitempty"`
}

// VSpherePlatformStatus holds the current status of the vSphere infrastructure provider.
type VSpherePlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	//
	// Deprecated: Use APIServerInternalIPs instead.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// apiServerInternalIPs are the IP addresses to contact the Kubernetes API
	// server that can be used by components inside the cluster, like kubelets
	// using the infrastructure rather than Kubernetes networking. These are the
	// IPs for a self-hosted load balancer in front of the API servers. In dual
	// stack clusters this list contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	APIServerInternalIPs []string `json:"apiServerInternalIPs"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	//
	// Deprecated: Use IngressIPs instead.
	IngressIP string `json:"ingressIP,omitempty"`

	// ingressIPs are the external IPs which route to the default ingress
	// controller. The IPs are suitable targets of a wildcard DNS record used to
	// resolve default route host names. In dual stack clusters this list
	// contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	IngressIPs []string `json:"ingressIPs"`

	// nodeDNSIP is the IP address for the internal DNS used by the
	// nodes. Unlike the one managed by the DNS operator, `NodeDNSIP`
	// provides name resolution for the nodes themselves. There is no DNS-as-a-service for
	// vSphere deployments. In order to minimize necessary changes to the
	// datacenter DNS, a DNS service is hosted as a static pod to serve those hostnames
	// to the nodes in the cluster.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`

	// loadBalancer defines how the load balancer used by the cluster is configured.
	// +default={"type": "OpenShiftManagedDefault"}
	// +kubebuilder:default={"type": "OpenShiftManagedDefault"}
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +optional
	LoadBalancer *VSpherePlatformLoadBalancer `json:"loadBalancer,omitempty"`
}

// IBMCloudServiceEndpoint stores the configuration of a custom url to
// override existing defaults of IBM Cloud Services.
type IBMCloudServiceEndpoint struct {
	// name is the name of the IBM Cloud service.
	// Possible values are: CIS, COS, DNSServices, GlobalSearch, GlobalTagging, HyperProtect, IAM, KeyProtect, ResourceController, ResourceManager, or VPC.
	// For example, the IBM Cloud Private IAM service could be configured with the
	// service `name` of `IAM` and `url` of `https://private.iam.cloud.ibm.com`
	// Whereas the IBM Cloud Private VPC service for US South (Dallas) could be configured
	// with the service `name` of `VPC` and `url` of `https://us.south.private.iaas.cloud.ibm.com`
	//
	// +kubebuilder:validation:Required
	Name IBMCloudServiceName `json:"name"`

	// url is fully qualified URI with scheme https, that overrides the default generated
	// endpoint for a client.
	// This must be provided and cannot be empty.
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Type=string
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="url must be a valid absolute URL"
	URL string `json:"url"`
}

// IBMCloudPlatformSpec holds the desired state of the IBMCloud infrastructure provider.
// This only includes fields that can be modified in the cluster.
type IBMCloudPlatformSpec struct{}

// IBMCloudPlatformStatus holds the current status of the IBMCloud infrastructure provider.
type IBMCloudPlatformStatus struct {
	// Location is where the cluster has been deployed
	Location string `json:"location,omitempty"`

	// ResourceGroupName is the Resource Group for new IBMCloud resources created for the cluster.
	ResourceGroupName string `json:"resourceGroupName,omitempty"`

	// ProviderType indicates the type of cluster that was created
	ProviderType IBMCloudProviderType `json:"providerType,omitempty"`

	// CISInstanceCRN is the CRN of the Cloud Internet Services instance managing
	// the DNS zone for the cluster's base domain
	CISInstanceCRN string `json:"cisInstanceCRN,omitempty"`

	// DNSInstanceCRN is the CRN of the DNS Services instance managing the DNS zone
	// for the cluster's base domain
	DNSInstanceCRN string `json:"dnsInstanceCRN,omitempty"`

	// serviceEndpoints is a list of custom endpoints which will override the default
	// service endpoints of an IBM Cloud service. These endpoints are consumed by
	// components within the cluster to reach the respective IBM Cloud Services.
	// +listType=map
	// +listMapKey=name
	// +optional
	ServiceEndpoints []IBMCloudServiceEndpoint `json:"serviceEndpoints,omitempty"`
}

// KubevirtPlatformSpec holds the desired state of the kubevirt infrastructure provider.
// This only includes fields that can be modified in the cluster.
type KubevirtPlatformSpec struct{}

// KubevirtPlatformStatus holds the current status of the kubevirt infrastructure provider.
type KubevirtPlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	IngressIP string `json:"ingressIP,omitempty"`
}

// EquinixMetalPlatformSpec holds the desired state of the Equinix Metal infrastructure provider.
// This only includes fields that can be modified in the cluster.
type EquinixMetalPlatformSpec struct{}

// EquinixMetalPlatformStatus holds the current status of the Equinix Metal infrastructure provider.
type EquinixMetalPlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	IngressIP string `json:"ingressIP,omitempty"`
}

// PowervsServiceEndpoint stores the configuration of a custom url to
// override existing defaults of PowerVS Services.
type PowerVSServiceEndpoint struct {
	// name is the name of the Power VS service.
	// Few of the services are
	// IAM - https://cloud.ibm.com/apidocs/iam-identity-token-api
	// ResourceController - https://cloud.ibm.com/apidocs/resource-controller/resource-controller
	// Power Cloud - https://cloud.ibm.com/apidocs/power-cloud
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^[a-z0-9-]+$`
	Name string `json:"name"`

	// url is fully qualified URI with scheme https, that overrides the default generated
	// endpoint for a client.
	// This must be provided and cannot be empty.
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Type=string
	// +kubebuilder:validation:Format=uri
	// +kubebuilder:validation:Pattern=`^https://`
	URL string `json:"url"`
}

// PowerVSPlatformSpec holds the desired state of the IBM Power Systems Virtual Servers infrastructure provider.
// This only includes fields that can be modified in the cluster.
type PowerVSPlatformSpec struct {
	// serviceEndpoints is a list of custom endpoints which will override the default
	// service endpoints of a Power VS service.
	// +listType=map
	// +listMapKey=name
	// +optional
	ServiceEndpoints []PowerVSServiceEndpoint `json:"serviceEndpoints,omitempty"`
}

// PowerVSPlatformStatus holds the current status of the IBM Power Systems Virtual Servers infrastrucutre provider.
// +kubebuilder:validation:XValidation:rule="!has(oldSelf.resourceGroup) || has(self.resourceGroup)",message="cannot unset resourceGroup once set"
type PowerVSPlatformStatus struct {
	// region holds the default Power VS region for new Power VS resources created by the cluster.
	Region string `json:"region"`

	// zone holds the default zone for the new Power VS resources created by the cluster.
	// Note: Currently only single-zone OCP clusters are supported
	Zone string `json:"zone"`

	// resourceGroup is the resource group name for new IBMCloud resources created for a cluster.
	// The resource group specified here will be used by cluster-image-registry-operator to set up a COS Instance in IBMCloud for the cluster registry.
	// More about resource groups can be found here: https://cloud.ibm.com/docs/account?topic=account-rgs.
	// When omitted, the image registry operator won't be able to configure storage,
	// which results in the image registry cluster operator not being in an available state.
	//
	// +kubebuilder:validation:Pattern=^[a-zA-Z0-9-_ ]+$
	// +kubebuilder:validation:MaxLength=40
	// +kubebuilder:validation:XValidation:rule="oldSelf == '' || self == oldSelf",message="resourceGroup is immutable once set"
	// +optional
	ResourceGroup string `json:"resourceGroup"`

	// serviceEndpoints is a list of custom endpoints which will override the default
	// service endpoints of a Power VS service.
	// +listType=map
	// +listMapKey=name
	// +optional
	ServiceEndpoints []PowerVSServiceEndpoint `json:"serviceEndpoints,omitempty"`

	// CISInstanceCRN is the CRN of the Cloud Internet Services instance managing
	// the DNS zone for the cluster's base domain
	CISInstanceCRN string `json:"cisInstanceCRN,omitempty"`

	// DNSInstanceCRN is the CRN of the DNS Services instance managing the DNS zone
	// for the cluster's base domain
	DNSInstanceCRN string `json:"dnsInstanceCRN,omitempty"`
}

// AlibabaCloudPlatformSpec holds the desired state of the Alibaba Cloud infrastructure provider.
// This only includes fields that can be modified in the cluster.
type AlibabaCloudPlatformSpec struct{}

// AlibabaCloudPlatformStatus holds the current status of the Alibaba Cloud infrastructure provider.
type AlibabaCloudPlatformStatus struct {
	// region specifies the region for Alibaba Cloud resources created for the cluster.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^[0-9A-Za-z-]+$`
	// +required
	Region string `json:"region"`
	// resourceGroupID is the ID of the resource group for the cluster.
	// +kubebuilder:validation:Pattern=`^(rg-[0-9A-Za-z]+)?$`
	// +optional
	ResourceGroupID string `json:"resourceGroupID,omitempty"`
	// resourceTags is a list of additional tags to apply to Alibaba Cloud resources created for the cluster.
	// +kubebuilder:validation:MaxItems=20
	// +listType=map
	// +listMapKey=key
	// +optional
	ResourceTags []AlibabaCloudResourceTag `json:"resourceTags,omitempty"`
}

// AlibabaCloudResourceTag is the set of tags to add to apply to resources.
type AlibabaCloudResourceTag struct {
	// key is the key of the tag.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	// +required
	Key string `json:"key"`
	// value is the value of the tag.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	// +required
	Value string `json:"value"`
}

// NutanixPlatformLoadBalancer defines the load balancer used by the cluster on Nutanix platform.
// +union
type NutanixPlatformLoadBalancer struct {
	// type defines the type of load balancer used by the cluster on Nutanix platform
	// which can be a user-managed or openshift-managed load balancer
	// that is to be used for the OpenShift API and Ingress endpoints.
	// When set to OpenShiftManagedDefault the static pods in charge of API and Ingress traffic load-balancing
	// defined in the machine config operator will be deployed.
	// When set to UserManaged these static pods will not be deployed and it is expected that
	// the load balancer is configured out of band by the deployer.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default.
	// The default value is OpenShiftManagedDefault.
	// +default="OpenShiftManagedDefault"
	// +kubebuilder:default:="OpenShiftManagedDefault"
	// +kubebuilder:validation:Enum:="OpenShiftManagedDefault";"UserManaged"
	// +kubebuilder:validation:XValidation:rule="oldSelf == '' || self == oldSelf",message="type is immutable once set"
	// +optional
	// +unionDiscriminator
	Type PlatformLoadBalancerType `json:"type,omitempty"`
}

// NutanixPlatformSpec holds the desired state of the Nutanix infrastructure provider.
// This only includes fields that can be modified in the cluster.
type NutanixPlatformSpec struct {
	// prismCentral holds the endpoint address and port to access the Nutanix Prism Central.
	// When a cluster-wide proxy is installed, by default, this endpoint will be accessed via the proxy.
	// Should you wish for communication with this endpoint not to be proxied, please add the endpoint to the
	// proxy spec.noProxy list.
	// +kubebuilder:validation:Required
	PrismCentral NutanixPrismEndpoint `json:"prismCentral"`

	// prismElements holds one or more endpoint address and port data to access the Nutanix
	// Prism Elements (clusters) of the Nutanix Prism Central. Currently we only support one
	// Prism Element (cluster) for an OpenShift cluster, where all the Nutanix resources (VMs, subnets, volumes, etc.)
	// used in the OpenShift cluster are located. In the future, we may support Nutanix resources (VMs, etc.)
	// spread over multiple Prism Elements (clusters) of the Prism Central.
	// +kubebuilder:validation:Required
	// +listType=map
	// +listMapKey=name
	PrismElements []NutanixPrismElementEndpoint `json:"prismElements"`

	// failureDomains configures failure domains information for the Nutanix platform.
	// When set, the failure domains defined here may be used to spread Machines across
	// prism element clusters to improve fault tolerance of the cluster.
	// +listType=map
	// +listMapKey=name
	// +optional
	FailureDomains []NutanixFailureDomain `json:"failureDomains"`
}

// NutanixFailureDomain configures failure domain information for the Nutanix platform.
type NutanixFailureDomain struct {
	// name defines the unique name of a failure domain.
	// Name is required and must be at most 64 characters in length.
	// It must consist of only lower case alphanumeric characters and hyphens (-).
	// It must start and end with an alphanumeric character.
	// This value is arbitrary and is used to identify the failure domain within the platform.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=64
	// +kubebuilder:validation:Pattern=`[a-z0-9]([-a-z0-9]*[a-z0-9])?`
	Name string `json:"name"`

	// cluster is to identify the cluster (the Prism Element under management of the Prism Central),
	// in which the Machine's VM will be created. The cluster identifier (uuid or name) can be obtained
	// from the Prism Central console or using the prism_central API.
	// +kubebuilder:validation:Required
	Cluster NutanixResourceIdentifier `json:"cluster"`

	// subnets holds a list of identifiers (one or more) of the cluster's network subnets
	// for the Machine's VM to connect to. The subnet identifiers (uuid or name) can be
	// obtained from the Prism Central console or using the prism_central API.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=1
	// +listType=map
	// +listMapKey=type
	Subnets []NutanixResourceIdentifier `json:"subnets"`
}

// NutanixIdentifierType is an enumeration of different resource identifier types.
// +kubebuilder:validation:Enum:=UUID;Name
type NutanixIdentifierType string

const (
	// NutanixIdentifierUUID is a resource identifier identifying the object by UUID.
	NutanixIdentifierUUID NutanixIdentifierType = "UUID"

	// NutanixIdentifierName is a resource identifier identifying the object by Name.
	NutanixIdentifierName NutanixIdentifierType = "Name"
)

// NutanixResourceIdentifier holds the identity of a Nutanix PC resource (cluster, image, subnet, etc.)
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'UUID' ?  has(self.uuid) : !has(self.uuid)",message="uuid configuration is required when type is UUID, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Name' ?  has(self.name) : !has(self.name)",message="name configuration is required when type is Name, and forbidden otherwise"
// +union
type NutanixResourceIdentifier struct {
	// type is the identifier type to use for this resource.
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	Type NutanixIdentifierType `json:"type"`

	// uuid is the UUID of the resource in the PC. It cannot be empty if the type is UUID.
	// +optional
	UUID *string `json:"uuid,omitempty"`

	// name is the resource name in the PC. It cannot be empty if the type is Name.
	// +optional
	Name *string `json:"name,omitempty"`
}

// NutanixPrismEndpoint holds the endpoint address and port to access the Nutanix Prism Central or Element (cluster)
type NutanixPrismEndpoint struct {
	// address is the endpoint address (DNS name or IP address) of the Nutanix Prism Central or Element (cluster)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=256
	Address string `json:"address"`

	// port is the port number to access the Nutanix Prism Central or Element (cluster)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	Port int32 `json:"port"`
}

// NutanixPrismElementEndpoint holds the name and endpoint data for a Prism Element (cluster)
type NutanixPrismElementEndpoint struct {
	// name is the name of the Prism Element (cluster). This value will correspond with
	// the cluster field configured on other resources (eg Machines, PVCs, etc).
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=256
	Name string `json:"name"`

	// endpoint holds the endpoint address and port data of the Prism Element (cluster).
	// When a cluster-wide proxy is installed, by default, this endpoint will be accessed via the proxy.
	// Should you wish for communication with this endpoint not to be proxied, please add the endpoint to the
	// proxy spec.noProxy list.
	// +kubebuilder:validation:Required
	Endpoint NutanixPrismEndpoint `json:"endpoint"`
}

// NutanixPlatformStatus holds the current status of the Nutanix infrastructure provider.
type NutanixPlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	//
	// Deprecated: Use APIServerInternalIPs instead.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// apiServerInternalIPs are the IP addresses to contact the Kubernetes API
	// server that can be used by components inside the cluster, like kubelets
	// using the infrastructure rather than Kubernetes networking. These are the
	// IPs for a self-hosted load balancer in front of the API servers. In dual
	// stack clusters this list contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	APIServerInternalIPs []string `json:"apiServerInternalIPs"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	//
	// Deprecated: Use IngressIPs instead.
	IngressIP string `json:"ingressIP,omitempty"`

	// ingressIPs are the external IPs which route to the default ingress
	// controller. The IPs are suitable targets of a wildcard DNS record used to
	// resolve default route host names. In dual stack clusters this list
	// contains two IPs otherwise only one.
	//
	// +kubebuilder:validation:Format=ip
	// +kubebuilder:validation:MaxItems=2
	// +listType=set
	IngressIPs []string `json:"ingressIPs"`

	// loadBalancer defines how the load balancer used by the cluster is configured.
	// +default={"type": "OpenShiftManagedDefault"}
	// +kubebuilder:default={"type": "OpenShiftManagedDefault"}
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +optional
	LoadBalancer *NutanixPlatformLoadBalancer `json:"loadBalancer,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InfrastructureList is
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type InfrastructureList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Infrastructure `json:"items"`
}

// IP is an IP address (for example, "10.0.0.0" or "fd00::").
// +kubebuilder:validation:Pattern=`(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)|(^s*((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:)))(%.+)?s*)`
// + ---
// + The regex for the IPv4 and IPv6 address was taken from
// + https://blog.markhatton.co.uk/2011/03/15/regular-expressions-for-ip-addresses-cidr-ranges-and-hostnames/
// + The resulting regex is an OR of both regexes.
type IP string
