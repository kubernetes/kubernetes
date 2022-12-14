package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:subresource:status

// Infrastructure holds cluster-wide information about Infrastructure.  The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
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

// PlatformType is a specific supported infrastructure provider.
// +kubebuilder:validation:Enum="";AWS;Azure;BareMetal;GCP;Libvirt;OpenStack;None;VSphere;oVirt;IBMCloud;KubeVirt;EquinixMetal;PowerVS;AlibabaCloud;Nutanix
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

	// resourceTags is a list of additional tags to apply to AWS resources created for the cluster.
	// See https://docs.aws.amazon.com/general/latest/gr/aws_tagging.html for information on tagging AWS resources.
	// AWS supports a maximum of 50 tags per resource. OpenShift reserves 25 tags for its use, leaving 25 tags
	// available for the user.
	// +kubebuilder:validation:MaxItems=25
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
	IngressIPs []string `json:"ingressIPs"`

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
	IngressIPs []string `json:"ingressIPs"`

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
	IngressIPs []string `json:"ingressIPs"`

	// deprecated: as of 4.6, this field is no longer set or honored.  It will be removed in a future release.
	NodeDNSIP string `json:"nodeDNSIP,omitempty"`
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
	// +openshift:enable:FeatureSets=TechPreviewNoUpgrade
	// +kubebuilder:validation:MaxItems=1
	// +kubebuilder:validation:MinItems=0
	// +optional
	VCenters []VSpherePlatformVCenterSpec `json:"vcenters,omitempty"`

	// failureDomains contains the definition of region, zone and the vCenter topology.
	// If this is omitted failure domains (regions and zones) will not be used.
	// +openshift:enable:FeatureSets=TechPreviewNoUpgrade
	// +optional
	FailureDomains []VSpherePlatformFailureDomainSpec `json:"failureDomains,omitempty"`

	// nodeNetworking contains the definition of internal and external network constraints for
	// assigning the node's networking.
	// If this field is omitted, networking defaults to the legacy
	// address selection behavior which is to only support a single address and
	// return the first one found.
	// +openshift:enable:FeatureSets=TechPreviewNoUpgrade
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
	IngressIPs []string `json:"ingressIPs"`

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
type PowerVSPlatformStatus struct {
	// region holds the default Power VS region for new Power VS resources created by the cluster.
	Region string `json:"region"`

	// zone holds the default zone for the new Power VS resources created by the cluster.
	// Note: Currently only single-zone OCP clusters are supported
	Zone string `json:"zone"`

	// serviceEndpoints is a list of custom endpoints which will override the default
	// service endpoints of a Power VS service.
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
	IngressIPs []string `json:"ingressIPs"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InfrastructureList is
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type InfrastructureList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Infrastructure `json:"items"`
}
