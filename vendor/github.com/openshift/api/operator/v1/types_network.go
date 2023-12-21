package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Network describes the cluster's desired network configuration. It is
// consumed by the cluster-network-operator.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +k8s:openapi-gen=true
// +openshift:compatibility-gen:level=1
type Network struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   NetworkSpec   `json:"spec,omitempty"`
	Status NetworkStatus `json:"status,omitempty"`
}

// NetworkStatus is detailed operator status, which is distilled
// up to the Network clusteroperator object.
type NetworkStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NetworkList contains a list of Network configurations
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type NetworkList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []Network `json:"items"`
}

// NetworkSpec is the top-level network configuration object.
type NetworkSpec struct {
	OperatorSpec `json:",inline"`

	// clusterNetwork is the IP address pool to use for pod IPs.
	// Some network providers, e.g. OpenShift SDN, support multiple ClusterNetworks.
	// Others only support one. This is equivalent to the cluster-cidr.
	ClusterNetwork []ClusterNetworkEntry `json:"clusterNetwork"`

	// serviceNetwork is the ip address pool to use for Service IPs
	// Currently, all existing network providers only support a single value
	// here, but this is an array to allow for growth.
	ServiceNetwork []string `json:"serviceNetwork"`

	// defaultNetwork is the "default" network that all pods will receive
	DefaultNetwork DefaultNetworkDefinition `json:"defaultNetwork"`

	// additionalNetworks is a list of extra networks to make available to pods
	// when multiple networks are enabled.
	AdditionalNetworks []AdditionalNetworkDefinition `json:"additionalNetworks,omitempty"`

	// disableMultiNetwork specifies whether or not multiple pod network
	// support should be disabled. If unset, this property defaults to
	// 'false' and multiple network support is enabled.
	DisableMultiNetwork *bool `json:"disableMultiNetwork,omitempty"`

	// useMultiNetworkPolicy enables a controller which allows for
	// MultiNetworkPolicy objects to be used on additional networks as
	// created by Multus CNI. MultiNetworkPolicy are similar to NetworkPolicy
	// objects, but NetworkPolicy objects only apply to the primary interface.
	// With MultiNetworkPolicy, you can control the traffic that a pod can receive
	// over the secondary interfaces. If unset, this property defaults to 'false'
	// and MultiNetworkPolicy objects are ignored. If 'disableMultiNetwork' is
	// 'true' then the value of this field is ignored.
	UseMultiNetworkPolicy *bool `json:"useMultiNetworkPolicy,omitempty"`

	// deployKubeProxy specifies whether or not a standalone kube-proxy should
	// be deployed by the operator. Some network providers include kube-proxy
	// or similar functionality. If unset, the plugin will attempt to select
	// the correct value, which is false when OpenShift SDN and ovn-kubernetes are
	// used and true otherwise.
	// +optional
	DeployKubeProxy *bool `json:"deployKubeProxy,omitempty"`

	// disableNetworkDiagnostics specifies whether or not PodNetworkConnectivityCheck
	// CRs from a test pod to every node, apiserver and LB should be disabled or not.
	// If unset, this property defaults to 'false' and network diagnostics is enabled.
	// Setting this to 'true' would reduce the additional load of the pods performing the checks.
	// +optional
	// +kubebuilder:default:=false
	DisableNetworkDiagnostics bool `json:"disableNetworkDiagnostics"`

	// kubeProxyConfig lets us configure desired proxy configuration.
	// If not specified, sensible defaults will be chosen by OpenShift directly.
	// Not consumed by all network providers - currently only openshift-sdn.
	KubeProxyConfig *ProxyConfig `json:"kubeProxyConfig,omitempty"`

	// exportNetworkFlows enables and configures the export of network flow metadata from the pod network
	// by using protocols NetFlow, SFlow or IPFIX. Currently only supported on OVN-Kubernetes plugin.
	// If unset, flows will not be exported to any collector.
	// +optional
	ExportNetworkFlows *ExportNetworkFlows `json:"exportNetworkFlows,omitempty"`

	// migration enables and configures the cluster network migration. The
	// migration procedure allows to change the network type and the MTU.
	// +optional
	Migration *NetworkMigration `json:"migration,omitempty"`
}

// NetworkMigrationMode is an enumeration of the possible mode of the network migration
// Valid values are "Live", "Offline" and omitted.
// +kubebuilder:validation:Enum:=Live;Offline;""
type NetworkMigrationMode string

const (
	// A "Live" migration operation will not cause service interruption by migrating the CNI of each node one by one. The cluster network will work as normal during the network migration.
	LiveNetworkMigrationMode NetworkMigrationMode = "Live"
	// An "Offline" migration operation will cause service interruption. During an "Offline" migration, two rounds of node reboots are required. The cluster network will be malfunctioning during the network migration.
	OfflineNetworkMigrationMode NetworkMigrationMode = "Offline"
)

// NetworkMigration represents the cluster network configuration.
// +openshift:validation:FeatureSetAwareXValidation:featureSet=CustomNoUpgrade;TechPreviewNoUpgrade,rule="!has(self.mtu) || !has(self.networkType) || self.networkType == '' || has(self.mode) && self.mode == 'Live'",message="networkType migration in mode other than 'Live' may not be configured at the same time as mtu migration"
type NetworkMigration struct {
	// networkType is the target type of network migration. Set this to the
	// target network type to allow changing the default network. If unset, the
	// operation of changing cluster default network plugin will be rejected.
	// The supported values are OpenShiftSDN, OVNKubernetes
	// +optional
	NetworkType string `json:"networkType,omitempty"`

	// mtu contains the MTU migration configuration. Set this to allow changing
	// the MTU values for the default network. If unset, the operation of
	// changing the MTU for the default network will be rejected.
	// +optional
	MTU *MTUMigration `json:"mtu,omitempty"`

	// features contains the features migration configuration. Set this to migrate
	// feature configuration when changing the cluster default network provider.
	// if unset, the default operation is to migrate all the configuration of
	// supported features.
	// +optional
	Features *FeaturesMigration `json:"features,omitempty"`

	// mode indicates the mode of network migration.
	// The supported values are "Live", "Offline" and omitted.
	// A "Live" migration operation will not cause service interruption by migrating the CNI of each node one by one. The cluster network will work as normal during the network migration.
	// An "Offline" migration operation will cause service interruption. During an "Offline" migration, two rounds of node reboots are required. The cluster network will be malfunctioning during the network migration.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default which is subject to change over time.
	// The current default value is "Offline".
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +optional
	Mode NetworkMigrationMode `json:"mode"`
}

type FeaturesMigration struct {
	// egressIP specifies whether or not the Egress IP configuration is migrated
	// automatically when changing the cluster default network provider.
	// If unset, this property defaults to 'true' and Egress IP configure is migrated.
	// +optional
	// +kubebuilder:default:=true
	EgressIP bool `json:"egressIP,omitempty"`
	// egressFirewall specifies whether or not the Egress Firewall configuration is migrated
	// automatically when changing the cluster default network provider.
	// If unset, this property defaults to 'true' and Egress Firewall configure is migrated.
	// +optional
	// +kubebuilder:default:=true
	EgressFirewall bool `json:"egressFirewall,omitempty"`
	// multicast specifies whether or not the multicast configuration is migrated
	// automatically when changing the cluster default network provider.
	// If unset, this property defaults to 'true' and multicast configure is migrated.
	// +optional
	// +kubebuilder:default:=true
	Multicast bool `json:"multicast,omitempty"`
}

// MTUMigration MTU contains infomation about MTU migration.
type MTUMigration struct {
	// network contains information about MTU migration for the default network.
	// Migrations are only allowed to MTU values lower than the machine's uplink
	// MTU by the minimum appropriate offset.
	// +optional
	Network *MTUMigrationValues `json:"network,omitempty"`

	// machine contains MTU migration configuration for the machine's uplink.
	// Needs to be migrated along with the default network MTU unless the
	// current uplink MTU already accommodates the default network MTU.
	// +optional
	Machine *MTUMigrationValues `json:"machine,omitempty"`
}

// MTUMigrationValues contains the values for a MTU migration.
type MTUMigrationValues struct {
	// to is the MTU to migrate to.
	// +kubebuilder:validation:Minimum=0
	To *uint32 `json:"to"`

	// from is the MTU to migrate from.
	// +kubebuilder:validation:Minimum=0
	// +optional
	From *uint32 `json:"from,omitempty"`
}

// ClusterNetworkEntry is a subnet from which to allocate PodIPs. A network of size
// HostPrefix (in CIDR notation) will be allocated when nodes join the cluster. If
// the HostPrefix field is not used by the plugin, it can be left unset.
// Not all network providers support multiple ClusterNetworks
type ClusterNetworkEntry struct {
	CIDR string `json:"cidr"`
	// +kubebuilder:validation:Minimum=0
	// +optional
	HostPrefix uint32 `json:"hostPrefix,omitempty"`
}

// DefaultNetworkDefinition represents a single network plugin's configuration.
// type must be specified, along with exactly one "Config" that matches the type.
type DefaultNetworkDefinition struct {
	// type is the type of network
	// All NetworkTypes are supported except for NetworkTypeRaw
	Type NetworkType `json:"type"`

	// openShiftSDNConfig configures the openshift-sdn plugin
	// +optional
	OpenShiftSDNConfig *OpenShiftSDNConfig `json:"openshiftSDNConfig,omitempty"`

	// ovnKubernetesConfig configures the ovn-kubernetes plugin.
	// +optional
	OVNKubernetesConfig *OVNKubernetesConfig `json:"ovnKubernetesConfig,omitempty"`
}

// SimpleMacvlanConfig contains configurations for macvlan interface.
type SimpleMacvlanConfig struct {
	// master is the host interface to create the macvlan interface from.
	// If not specified, it will be default route interface
	// +optional
	Master string `json:"master,omitempty"`

	// IPAMConfig configures IPAM module will be used for IP Address Management (IPAM).
	// +optional
	IPAMConfig *IPAMConfig `json:"ipamConfig,omitempty"`

	// mode is the macvlan mode: bridge, private, vepa, passthru. The default is bridge
	// +optional
	Mode MacvlanMode `json:"mode,omitempty"`

	// mtu is the mtu to use for the macvlan interface. if unset, host's
	// kernel will select the value.
	// +kubebuilder:validation:Minimum=0
	// +optional
	MTU uint32 `json:"mtu,omitempty"`
}

// StaticIPAMAddresses provides IP address and Gateway for static IPAM addresses
type StaticIPAMAddresses struct {
	// Address is the IP address in CIDR format
	// +optional
	Address string `json:"address"`
	// Gateway is IP inside of subnet to designate as the gateway
	// +optional
	Gateway string `json:"gateway,omitempty"`
}

// StaticIPAMRoutes provides Destination/Gateway pairs for static IPAM routes
type StaticIPAMRoutes struct {
	// Destination points the IP route destination
	Destination string `json:"destination"`
	// Gateway is the route's next-hop IP address
	// If unset, a default gateway is assumed (as determined by the CNI plugin).
	// +optional
	Gateway string `json:"gateway,omitempty"`
}

// StaticIPAMDNS provides DNS related information for static IPAM
type StaticIPAMDNS struct {
	// Nameservers points DNS servers for IP lookup
	// +optional
	Nameservers []string `json:"nameservers,omitempty"`
	// Domain configures the domainname the local domain used for short hostname lookups
	// +optional
	Domain string `json:"domain,omitempty"`
	// Search configures priority ordered search domains for short hostname lookups
	// +optional
	Search []string `json:"search,omitempty"`
}

// StaticIPAMConfig contains configurations for static IPAM (IP Address Management)
type StaticIPAMConfig struct {
	// Addresses configures IP address for the interface
	// +optional
	Addresses []StaticIPAMAddresses `json:"addresses,omitempty"`
	// Routes configures IP routes for the interface
	// +optional
	Routes []StaticIPAMRoutes `json:"routes,omitempty"`
	// DNS configures DNS for the interface
	// +optional
	DNS *StaticIPAMDNS `json:"dns,omitempty"`
}

// IPAMConfig contains configurations for IPAM (IP Address Management)
type IPAMConfig struct {
	// Type is the type of IPAM module will be used for IP Address Management(IPAM).
	// The supported values are IPAMTypeDHCP, IPAMTypeStatic
	Type IPAMType `json:"type"`

	// StaticIPAMConfig configures the static IP address in case of type:IPAMTypeStatic
	// +optional
	StaticIPAMConfig *StaticIPAMConfig `json:"staticIPAMConfig,omitempty"`
}

// AdditionalNetworkDefinition configures an extra network that is available but not
// created by default. Instead, pods must request them by name.
// type must be specified, along with exactly one "Config" that matches the type.
type AdditionalNetworkDefinition struct {
	// type is the type of network
	// The supported values are NetworkTypeRaw, NetworkTypeSimpleMacvlan
	Type NetworkType `json:"type"`

	// name is the name of the network. This will be populated in the resulting CRD
	// This must be unique.
	Name string `json:"name"`

	// namespace is the namespace of the network. This will be populated in the resulting CRD
	// If not given the network will be created in the default namespace.
	Namespace string `json:"namespace,omitempty"`

	// rawCNIConfig is the raw CNI configuration json to create in the
	// NetworkAttachmentDefinition CRD
	RawCNIConfig string `json:"rawCNIConfig,omitempty"`

	// SimpleMacvlanConfig configures the macvlan interface in case of type:NetworkTypeSimpleMacvlan
	// +optional
	SimpleMacvlanConfig *SimpleMacvlanConfig `json:"simpleMacvlanConfig,omitempty"`
}

// OpenShiftSDNConfig configures the three openshift-sdn plugins
type OpenShiftSDNConfig struct {
	// mode is one of "Multitenant", "Subnet", or "NetworkPolicy"
	Mode SDNMode `json:"mode"`

	// vxlanPort is the port to use for all vxlan packets. The default is 4789.
	// +kubebuilder:validation:Minimum=0
	// +optional
	VXLANPort *uint32 `json:"vxlanPort,omitempty"`

	// mtu is the mtu to use for the tunnel interface. Defaults to 1450 if unset.
	// This must be 50 bytes smaller than the machine's uplink.
	// +kubebuilder:validation:Minimum=0
	// +optional
	MTU *uint32 `json:"mtu,omitempty"`

	// useExternalOpenvswitch used to control whether the operator would deploy an OVS
	// DaemonSet itself or expect someone else to start OVS. As of 4.6, OVS is always
	// run as a system service, and this flag is ignored.
	// DEPRECATED: non-functional as of 4.6
	// +optional
	UseExternalOpenvswitch *bool `json:"useExternalOpenvswitch,omitempty"`

	// enableUnidling controls whether or not the service proxy will support idling
	// and unidling of services. By default, unidling is enabled.
	EnableUnidling *bool `json:"enableUnidling,omitempty"`
}

// ovnKubernetesConfig contains the configuration parameters for networks
// using the ovn-kubernetes network project
type OVNKubernetesConfig struct {
	// mtu is the MTU to use for the tunnel interface. This must be 100
	// bytes smaller than the uplink mtu.
	// Default is 1400
	// +kubebuilder:validation:Minimum=0
	// +optional
	MTU *uint32 `json:"mtu,omitempty"`
	// geneve port is the UDP port to be used by geneve encapulation.
	// Default is 6081
	// +kubebuilder:validation:Minimum=1
	// +optional
	GenevePort *uint32 `json:"genevePort,omitempty"`
	// HybridOverlayConfig configures an additional overlay network for peers that are
	// not using OVN.
	// +optional
	HybridOverlayConfig *HybridOverlayConfig `json:"hybridOverlayConfig,omitempty"`
	// ipsecConfig enables and configures IPsec for pods on the pod network within the
	// cluster.
	// +optional
	IPsecConfig *IPsecConfig `json:"ipsecConfig,omitempty"`
	// policyAuditConfig is the configuration for network policy audit events. If unset,
	// reported defaults are used.
	// +optional
	PolicyAuditConfig *PolicyAuditConfig `json:"policyAuditConfig,omitempty"`
	// gatewayConfig holds the configuration for node gateway options.
	// +optional
	GatewayConfig *GatewayConfig `json:"gatewayConfig,omitempty"`
	// v4InternalSubnet is a v4 subnet used internally by ovn-kubernetes in case the
	// default one is being already used by something else. It must not overlap with
	// any other subnet being used by OpenShift or by the node network. The size of the
	// subnet must be larger than the number of nodes. The value cannot be changed
	// after installation.
	// Default is 100.64.0.0/16
	// +optional
	V4InternalSubnet string `json:"v4InternalSubnet,omitempty"`
	// v6InternalSubnet is a v6 subnet used internally by ovn-kubernetes in case the
	// default one is being already used by something else. It must not overlap with
	// any other subnet being used by OpenShift or by the node network. The size of the
	// subnet must be larger than the number of nodes. The value cannot be changed
	// after installation.
	// Default is fd98::/48
	// +optional
	V6InternalSubnet string `json:"v6InternalSubnet,omitempty"`
	// egressIPConfig holds the configuration for EgressIP options.
	// +optional
	EgressIPConfig EgressIPConfig `json:"egressIPConfig,omitempty"`
}

type HybridOverlayConfig struct {
	// HybridClusterNetwork defines a network space given to nodes on an additional overlay network.
	HybridClusterNetwork []ClusterNetworkEntry `json:"hybridClusterNetwork"`
	// HybridOverlayVXLANPort defines the VXLAN port number to be used by the additional overlay network.
	// Default is 4789
	// +optional
	HybridOverlayVXLANPort *uint32 `json:"hybridOverlayVXLANPort,omitempty"`
}

type IPsecConfig struct {
}

type IPForwardingMode string

const (
	// IPForwardingRestricted limits the IP forwarding on OVN-Kube managed interfaces (br-ex, br-ex1) to only required
	// service and other k8s related traffic
	IPForwardingRestricted IPForwardingMode = "Restricted"

	// IPForwardingGlobal allows all IP traffic to be forwarded across OVN-Kube managed interfaces
	IPForwardingGlobal IPForwardingMode = "Global"
)

// GatewayConfig holds node gateway-related parsed config file parameters and command-line overrides
type GatewayConfig struct {
	// RoutingViaHost allows pod egress traffic to exit via the ovn-k8s-mp0 management port
	// into the host before sending it out. If this is not set, traffic will always egress directly
	// from OVN to outside without touching the host stack. Setting this to true means hardware
	// offload will not be supported. Default is false if GatewayConfig is specified.
	// +kubebuilder:default:=false
	// +optional
	RoutingViaHost bool `json:"routingViaHost,omitempty"`
	// IPForwarding controls IP forwarding for all traffic on OVN-Kubernetes managed interfaces (such as br-ex).
	// By default this is set to Restricted, and Kubernetes related traffic is still forwarded appropriately, but other
	// IP traffic will not be routed by the OCP node. If there is a desire to allow the host to forward traffic across
	// OVN-Kubernetes managed interfaces, then set this field to "Global".
	// The supported values are "Restricted" and "Global".
	// +optional
	IPForwarding IPForwardingMode `json:"ipForwarding,omitempty"`
	// ipv4 allows users to configure IP settings for IPv4 connections. When omitted, this means no opinion and the default
	// configuration is used. Check individual members fields within ipv4 for details of default values.
	// +optional
	IPv4 IPv4GatewayConfig `json:"ipv4,omitempty"`
	// ipv6 allows users to configure IP settings for IPv6 connections. When omitted, this means no opinion and the default
	// configuration is used. Check individual members fields within ipv6 for details of default values.
	// +optional
	IPv6 IPv6GatewayConfig `json:"ipv6,omitempty"`
}

// IPV4GatewayConfig holds the configuration paramaters for IPV4 connections in the GatewayConfig for OVN-Kubernetes
type IPv4GatewayConfig struct {
	// internalMasqueradeSubnet contains the masquerade addresses in IPV4 CIDR format used internally by
	// ovn-kubernetes to enable host to service traffic. Each host in the cluster is configured with these
	// addresses, as well as the shared gateway bridge interface. The values can be changed after
	// installation. The subnet chosen should not overlap with other networks specified for
	// OVN-Kubernetes as well as other networks used on the host. Additionally the subnet must
	// be large enough to accommodate 6 IPs (maximum prefix length /29).
	// When omitted, this means no opinion and the platform is left to choose a reasonable default which is subject to change over time.
	// The current default subnet is 169.254.169.0/29
	// The value must be in proper IPV4 CIDR format
	// +kubebuilder:validation:MaxLength=18
	// +kubebuilder:validation:XValidation:rule="self.indexOf('/') == self.lastIndexOf('/')",message="CIDR format must contain exactly one '/'"
	// +kubebuilder:validation:XValidation:rule="[int(self.split('/')[1])].all(x, x <= 29 && x >= 0)",message="subnet must be in the range /0 to /29 inclusive"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split('.').size() == 4",message="a valid IPv4 address must contain 4 octets"
	// +kubebuilder:validation:XValidation:rule="[self.findAll('[0-9]+')[0]].all(x, x != '0' && int(x) <= 255 && !x.startsWith('0'))",message="first IP address octet must not contain leading zeros, must be greater than 0 and less or equal to 255"
	// +kubebuilder:validation:XValidation:rule="[self.findAll('[0-9]+')[1], self.findAll('[0-9]+')[2], self.findAll('[0-9]+')[3]].all(x, int(x) <= 255 && (x == '0' || !x.startsWith('0')))",message="IP address octets must not contain leading zeros, and must be less or equal to 255"
	// +optional
	InternalMasqueradeSubnet string `json:"internalMasqueradeSubnet,omitempty"`
}

// IPV6GatewayConfig holds the configuration paramaters for IPV6 connections in the GatewayConfig for OVN-Kubernetes
type IPv6GatewayConfig struct {
	// internalMasqueradeSubnet contains the masquerade addresses in IPV6 CIDR format used internally by
	// ovn-kubernetes to enable host to service traffic. Each host in the cluster is configured with these
	// addresses, as well as the shared gateway bridge interface. The values can be changed after
	// installation. The subnet chosen should not overlap with other networks specified for
	// OVN-Kubernetes as well as other networks used on the host. Additionally the subnet must
	// be large enough to accommodate 6 IPs (maximum prefix length /125).
	// When omitted, this means no opinion and the platform is left to choose a reasonable default which is subject to change over time.
	// The current default subnet is fd69::/125
	// Note that IPV6 dual addresses are not permitted
	// +kubebuilder:validation:XValidation:rule="self.indexOf('/') == self.lastIndexOf('/')",message="CIDR format must contain exactly one '/'"
	// +kubebuilder:validation:XValidation:rule="self.split('/').size() == 2 && [int(self.split('/')[1])].all(x, x <= 125 && x >= 0)",message="subnet must be in the range /0 to /125 inclusive"
	// +kubebuilder:validation:XValidation:rule="self.indexOf('::') == self.lastIndexOf('::')",message="IPv6 addresses must contain at most one '::' and may only be shortened once"
	// +kubebuilder:validation:XValidation:rule="self.contains('::') ? self.split('/')[0].split(':').size() <= 8 : self.split('/')[0].split(':').size() == 8",message="a valid IPv6 address must contain 8 segments unless elided (::), in which case it must contain at most 6 non-empty segments"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=1 ? [self.split('/')[0].split(':', 8)[0]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 1"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=2 ? [self.split('/')[0].split(':', 8)[1]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 2"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=3 ? [self.split('/')[0].split(':', 8)[2]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 3"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=4 ? [self.split('/')[0].split(':', 8)[3]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 4"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=5 ? [self.split('/')[0].split(':', 8)[4]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 5"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=6 ? [self.split('/')[0].split(':', 8)[5]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 6"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=7 ? [self.split('/')[0].split(':', 8)[6]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 7"
	// +kubebuilder:validation:XValidation:rule="self.split('/')[0].split(':').size() >=8 ? [self.split('/')[0].split(':', 8)[7]].all(x, x == '' || (x.matches('^[0-9A-Fa-f]{1,4}$')) && size(x)<5 ) : true",message="each segment of an IPv6 address must be a hexadecimal number between 0 and FFFF, failed on segment 8"
	// +kubebuilder:validation:XValidation:rule="!self.contains('.')",message="IPv6 dual addresses are not permitted, value should not contain `.` characters"
	// +optional
	InternalMasqueradeSubnet string `json:"internalMasqueradeSubnet,omitempty"`
}

type ExportNetworkFlows struct {
	// netFlow defines the NetFlow configuration.
	// +optional
	NetFlow *NetFlowConfig `json:"netFlow,omitempty"`
	// sFlow defines the SFlow configuration.
	// +optional
	SFlow *SFlowConfig `json:"sFlow,omitempty"`
	// ipfix defines IPFIX configuration.
	// +optional
	IPFIX *IPFIXConfig `json:"ipfix,omitempty"`
}

type NetFlowConfig struct {
	// netFlow defines the NetFlow collectors that will consume the flow data exported from OVS.
	// It is a list of strings formatted as ip:port with a maximum of ten items
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	Collectors []IPPort `json:"collectors,omitempty"`
}

type SFlowConfig struct {
	// sFlowCollectors is list of strings formatted as ip:port with a maximum of ten items
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	Collectors []IPPort `json:"collectors,omitempty"`
}

type IPFIXConfig struct {
	// ipfixCollectors is list of strings formatted as ip:port with a maximum of ten items
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	Collectors []IPPort `json:"collectors,omitempty"`
}

// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]):([1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$`
type IPPort string

type PolicyAuditConfig struct {
	// rateLimit is the approximate maximum number of messages to generate per-second per-node. If
	// unset the default of 20 msg/sec is used.
	// +kubebuilder:default=20
	// +kubebuilder:validation:Minimum=1
	// +optional
	RateLimit *uint32 `json:"rateLimit,omitempty"`

	// maxFilesSize is the max size an ACL_audit log file is allowed to reach before rotation occurs
	// Units are in MB and the Default is 50MB
	// +kubebuilder:default=50
	// +kubebuilder:validation:Minimum=1
	// +optional
	MaxFileSize *uint32 `json:"maxFileSize,omitempty"`

	// maxLogFiles specifies the maximum number of ACL_audit log files that can be present.
	// +kubebuilder:default=5
	// +kubebuilder:validation:Minimum=1
	// +optional
	MaxLogFiles *int32 `json:"maxLogFiles,omitempty"`

	// destination is the location for policy log messages.
	// Regardless of this config, persistent logs will always be dumped to the host
	// at /var/log/ovn/ however
	// Additionally syslog output may be configured as follows.
	// Valid values are:
	// - "libc" -> to use the libc syslog() function of the host node's journdald process
	// - "udp:host:port" -> for sending syslog over UDP
	// - "unix:file" -> for using the UNIX domain socket directly
	// - "null" -> to discard all messages logged to syslog
	// The default is "null"
	// +kubebuilder:default=null
	// +kubebuilder:pattern='^libc$|^null$|^udp:(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]):([0-9]){0,5}$|^unix:(\/[^\/ ]*)+([^\/\s])$'
	// +optional
	Destination string `json:"destination,omitempty"`

	// syslogFacility the RFC5424 facility for generated messages, e.g. "kern". Default is "local0"
	// +kubebuilder:default=local0
	// +kubebuilder:Enum=kern;user;mail;daemon;auth;syslog;lpr;news;uucp;clock;ftp;ntp;audit;alert;clock2;local0;local1;local2;local3;local4;local5;local6;local7
	// +optional
	SyslogFacility string `json:"syslogFacility,omitempty"`
}

// NetworkType describes the network plugin type to configure
type NetworkType string

// ProxyArgumentList is a list of arguments to pass to the kubeproxy process
type ProxyArgumentList []string

// ProxyConfig defines the configuration knobs for kubeproxy
// All of these are optional and have sensible defaults
type ProxyConfig struct {
	// An internal kube-proxy parameter. In older releases of OCP, this sometimes needed to be adjusted
	// in large clusters for performance reasons, but this is no longer necessary, and there is no reason
	// to change this from the default value.
	// Default: 30s
	IptablesSyncPeriod string `json:"iptablesSyncPeriod,omitempty"`

	// The address to "bind" on
	// Defaults to 0.0.0.0
	BindAddress string `json:"bindAddress,omitempty"`

	// Any additional arguments to pass to the kubeproxy process
	ProxyArguments map[string]ProxyArgumentList `json:"proxyArguments,omitempty"`
}

// EgressIPConfig defines the configuration knobs for egressip
type EgressIPConfig struct {
	// reachabilityTotalTimeout configures the EgressIP node reachability check total timeout in seconds.
	// If the EgressIP node cannot be reached within this timeout, the node is declared down.
	// Setting a large value may cause the EgressIP feature to react slowly to node changes.
	// In particular, it may react slowly for EgressIP nodes that really have a genuine problem and are unreachable.
	// When omitted, this means the user has no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The current default is 1 second.
	// A value of 0 disables the EgressIP node's reachability check.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=60
	// +optional
	ReachabilityTotalTimeoutSeconds *uint32 `json:"reachabilityTotalTimeoutSeconds,omitempty"`
}

const (
	// NetworkTypeOpenShiftSDN means the openshift-sdn plugin will be configured
	NetworkTypeOpenShiftSDN NetworkType = "OpenShiftSDN"

	// NetworkTypeOVNKubernetes means the ovn-kubernetes project will be configured.
	// This is currently not implemented.
	NetworkTypeOVNKubernetes NetworkType = "OVNKubernetes"

	// NetworkTypeRaw
	NetworkTypeRaw NetworkType = "Raw"

	// NetworkTypeSimpleMacvlan
	NetworkTypeSimpleMacvlan NetworkType = "SimpleMacvlan"
)

// SDNMode is the Mode the openshift-sdn plugin is in
type SDNMode string

const (
	// SDNModeSubnet is a simple mode that offers no isolation between pods
	SDNModeSubnet SDNMode = "Subnet"

	// SDNModeMultitenant is a special "multitenant" mode that offers limited
	// isolation configuration between namespaces
	SDNModeMultitenant SDNMode = "Multitenant"

	// SDNModeNetworkPolicy is a full NetworkPolicy implementation that allows
	// for sophisticated network isolation and segmenting. This is the default.
	SDNModeNetworkPolicy SDNMode = "NetworkPolicy"
)

// MacvlanMode is the Mode of macvlan. The value are lowercase to match the CNI plugin
// config values. See "man ip-link" for its detail.
type MacvlanMode string

const (
	// MacvlanModeBridge is the macvlan with thin bridge function.
	MacvlanModeBridge MacvlanMode = "Bridge"
	// MacvlanModePrivate
	MacvlanModePrivate MacvlanMode = "Private"
	// MacvlanModeVEPA is used with Virtual Ethernet Port Aggregator
	// (802.1qbg) swtich
	MacvlanModeVEPA MacvlanMode = "VEPA"
	// MacvlanModePassthru
	MacvlanModePassthru MacvlanMode = "Passthru"
)

// IPAMType describes the IP address management type to configure
type IPAMType string

const (
	// IPAMTypeDHCP uses DHCP for IP management
	IPAMTypeDHCP IPAMType = "DHCP"
	// IPAMTypeStatic uses static IP
	IPAMTypeStatic IPAMType = "Static"
)
