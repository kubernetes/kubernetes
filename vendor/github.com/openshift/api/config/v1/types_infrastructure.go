package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:subresource:status

// Infrastructure holds cluster-wide information about Infrastructure.  The canonical name is `cluster`
type Infrastructure struct {
	metav1.TypeMeta   `json:",inline"`
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
	// +kubebuilder:default=HighlyAvailable
	ControlPlaneTopology TopologyMode `json:"controlPlaneTopology"`

	// infrastructureTopology expresses the expectations for infrastructure services that do not run on control
	// plane nodes, usually indicated by a node selector for a `role` value
	// other than `master`.
	// The default is 'HighlyAvailable', which represents the behavior operators have in a "normal" cluster.
	// The 'SingleReplica' mode will be used in single-node deployments
	// and the operators should not configure the operand for highly-available operation
	// +kubebuilder:default=HighlyAvailable
	InfrastructureTopology TopologyMode `json:"infrastructureTopology"`
}

// TopologyMode defines the topology mode of the control/infra nodes.
// +kubebuilder:validation:Enum=HighlyAvailable;SingleReplica
type TopologyMode string

const (
	// "HighlyAvailable" is for operators to configure high-availability as much as possible.
	HighlyAvailableTopologyMode TopologyMode = "HighlyAvailable"

	// "SingleReplica" is for operators to avoid spending resources for high-availability purpose.
	SingleReplicaTopologyMode TopologyMode = "SingleReplica"
)

// PlatformType is a specific supported infrastructure provider.
// +kubebuilder:validation:Enum="";AWS;Azure;BareMetal;GCP;Libvirt;OpenStack;None;VSphere;oVirt;IBMCloud;KubeVirt;EquinixMetal
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
)

// IBMCloudProviderType is a specific supported IBM Cloud provider cluster type
type IBMCloudProviderType string

const (
	// Classic  means that the IBM Cloud cluster is using classic infrastructure
	IBMCloudProviderTypeClassic IBMCloudProviderType = "Classic"

	// VPC means that the IBM Cloud cluster is using VPC infrastructure
	IBMCloudProviderTypeVPC IBMCloudProviderType = "VPC"
)

// PlatformSpec holds the desired state specific to the underlying infrastructure provider
// of the current cluster. Since these are used at spec-level for the underlying cluster, it
// is supposed that only one of the spec structs is set.
type PlatformSpec struct {
	// type is the underlying infrastructure provider for the cluster. This
	// value controls whether infrastructure automation such as service load
	// balancers, dynamic volume provisioning, machine creation and deletion, and
	// other integrations are enabled. If None, no infrastructure automation is
	// enabled. Allowed values are "AWS", "Azure", "BareMetal", "GCP", "Libvirt",
	// "OpenStack", "VSphere", "oVirt", "KubeVirt", "EquinixMetal", and "None". Individual components may not support
	// all platforms, and must handle unrecognized platforms as None if they do
	// not support that platform.
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
	// "OpenStack", "VSphere", "oVirt", "EquinixMetal", and "None". Individual components may not support
	// all platforms, and must handle unrecognized platforms as None if they do
	// not support that platform.
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
	// +optional
	ServiceEndpoints []AWSServiceEndpoint `json:"serviceEndpoints,omitempty"`
}

// AzurePlatformSpec holds the desired state of the Azure infrastructure provider.
// This only includes fields that can be modified in the cluster.
type AzurePlatformSpec struct{}

// AzurePlatformStatus holds the current status of the Azure infrastructure provider.
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
}

// AzureCloudEnvironment is the name of the Azure cloud environment
// +kubebuilder:validation:Enum="";AzurePublicCloud;AzureUSGovernmentCloud;AzureChinaCloud;AzureGermanCloud
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
)

// GCPPlatformSpec holds the desired state of the Google Cloud Platform infrastructure provider.
// This only includes fields that can be modified in the cluster.
type GCPPlatformSpec struct{}

// GCPPlatformStatus holds the current status of the Google Cloud Platform infrastructure provider.
type GCPPlatformStatus struct {
	// resourceGroupName is the Project ID for new GCP resources created for the cluster.
	ProjectID string `json:"projectID"`

	// region holds the region for new GCP resources created for the cluster.
	Region string `json:"region"`
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
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	IngressIP string `json:"ingressIP,omitempty"`

	// nodeDNSIP is the IP address for the internal DNS used by the
	// nodes. Unlike the one managed by the DNS operator, `NodeDNSIP`
	// provides name resolution for the nodes themselves. There is no DNS-as-a-service for
	// BareMetal deployments. In order to minimize necessary changes to the
	// datacenter DNS, a DNS service is hosted as a static pod to serve those hostnames
	// to the nodes in the cluster.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`
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
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// cloudName is the name of the desired OpenStack cloud in the
	// client configuration file (`clouds.yaml`).
	CloudName string `json:"cloudName,omitempty"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	IngressIP string `json:"ingressIP,omitempty"`

	// nodeDNSIP is the IP address for the internal DNS used by the
	// nodes. Unlike the one managed by the DNS operator, `NodeDNSIP`
	// provides name resolution for the nodes themselves. There is no DNS-as-a-service for
	// OpenStack deployments. In order to minimize necessary changes to the
	// datacenter DNS, a DNS service is hosted as a static pod to serve those hostnames
	// to the nodes in the cluster.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`
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
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	IngressIP string `json:"ingressIP,omitempty"`

	// deprecated: as of 4.6, this field is no longer set or honored.  It will be removed in a future release.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`
}

// VSpherePlatformSpec holds the desired state of the vSphere infrastructure provider.
// This only includes fields that can be modified in the cluster.
type VSpherePlatformSpec struct{}

// VSpherePlatformStatus holds the current status of the vSphere infrastructure provider.
type VSpherePlatformStatus struct {
	// apiServerInternalIP is an IP address to contact the Kubernetes API server that can be used
	// by components inside the cluster, like kubelets using the infrastructure rather
	// than Kubernetes networking. It is the IP that the Infrastructure.status.apiServerInternalURI
	// points to. It is the IP for a self-hosted load balancer in front of the API servers.
	APIServerInternalIP string `json:"apiServerInternalIP,omitempty"`

	// ingressIP is an external IP which routes to the default ingress controller.
	// The IP is a suitable target of a wildcard DNS record used to resolve default route host names.
	IngressIP string `json:"ingressIP,omitempty"`

	// nodeDNSIP is the IP address for the internal DNS used by the
	// nodes. Unlike the one managed by the DNS operator, `NodeDNSIP`
	// provides name resolution for the nodes themselves. There is no DNS-as-a-service for
	// vSphere deployments. In order to minimize necessary changes to the
	// datacenter DNS, a DNS service is hosted as a static pod to serve those hostnames
	// to the nodes in the cluster.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`
}

// IBMCloudPlatformSpec holds the desired state of the IBMCloud infrastructure provider.
// This only includes fields that can be modified in the cluster.
type IBMCloudPlatformSpec struct{}

//IBMCloudPlatformStatus holds the current status of the IBMCloud infrastructure provider.
type IBMCloudPlatformStatus struct {
	// Location is where the cluster has been deployed
	Location string `json:"location,omitempty"`

	// ResourceGroupName is the Resource Group for new IBMCloud resources created for the cluster.
	ResourceGroupName string `json:"resourceGroupName,omitempty"`

	// ProviderType indicates the type of cluster that was created
	ProviderType IBMCloudProviderType `json:"providerType,omitempty"`
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InfrastructureList is
type InfrastructureList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Infrastructure `json:"items"`
}
