package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	ClusterNetworkDefault = "default"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterNetwork describes the cluster network. There is normally only one object of this type,
// named "default", which is created by the SDN network plugin based on the master configuration
// when the cluster is brought up for the first time.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +kubebuilder:resource:scope="Cluster"
// +kubebuilder:printcolumn:name="Cluster Network",type=string,JSONPath=`.network`,description="The primary cluster network CIDR"
// +kubebuilder:printcolumn:name="Service Network",type=string,JSONPath=`.serviceNetwork`,description="The service network CIDR"
// +kubebuilder:printcolumn:name="Plugin Name",type=string,JSONPath=`.pluginName`,description="The Openshift SDN network plug-in in use"
// +openshift:compatibility-gen:level=1
type ClusterNetwork struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Network is a CIDR string specifying the global overlay network's L3 space
	// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])/([0-9]|[12][0-9]|3[0-2])$`
	Network string `json:"network,omitempty" protobuf:"bytes,2,opt,name=network"`

	// HostSubnetLength is the number of bits of network to allocate to each node. eg, 8 would mean that each node would have a /24 slice of the overlay network for its pods
	// +kubebuilder:validation:Minimum=2
	// +kubebuilder:validation:Maximum=30
	HostSubnetLength uint32 `json:"hostsubnetlength,omitempty" protobuf:"varint,3,opt,name=hostsubnetlength"`

	// ServiceNetwork is the CIDR range that Service IP addresses are allocated from
	// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])/([0-9]|[12][0-9]|3[0-2])$`
	ServiceNetwork string `json:"serviceNetwork" protobuf:"bytes,4,opt,name=serviceNetwork"`

	// PluginName is the name of the network plugin being used
	PluginName string `json:"pluginName,omitempty" protobuf:"bytes,5,opt,name=pluginName"`

	// ClusterNetworks is a list of ClusterNetwork objects that defines the global overlay network's L3 space by specifying a set of CIDR and netmasks that the SDN can allocate addresses from.
	ClusterNetworks []ClusterNetworkEntry `json:"clusterNetworks" protobuf:"bytes,6,rep,name=clusterNetworks"`

	// VXLANPort sets the VXLAN destination port used by the cluster.
	// It is set by the master configuration file on startup and cannot be edited manually.
	// Valid values for VXLANPort are integers 1-65535 inclusive and if unset defaults to 4789.
	// Changing VXLANPort allows users to resolve issues between openshift SDN and other software trying to use the same VXLAN destination port.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:validation:Optional
	// +optional
	VXLANPort *uint32 `json:"vxlanPort,omitempty" protobuf:"varint,7,opt,name=vxlanPort"`

	// MTU is the MTU for the overlay network. This should be 50 less than the MTU of the network connecting the nodes. It is normally autodetected by the cluster network operator.
	// +kubebuilder:validation:Minimum=576
	// +kubebuilder:validation:Maximum=65536
	// +kubebuilder:validation:Optional
	// +optional
	MTU *uint32 `json:"mtu,omitempty" protobuf:"varint,8,opt,name=mtu"`
}

// ClusterNetworkEntry defines an individual cluster network. The CIDRs cannot overlap with other cluster network CIDRs, CIDRs reserved for external ips, CIDRs reserved for service networks, and CIDRs reserved for ingress ips.
type ClusterNetworkEntry struct {
	// CIDR defines the total range of a cluster networks address space.
	// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])/([0-9]|[12][0-9]|3[0-2])$`
	CIDR string `json:"CIDR" protobuf:"bytes,1,opt,name=cidr"`

	// HostSubnetLength is the number of bits of the accompanying CIDR address to allocate to each node. eg, 8 would mean that each node would have a /24 slice of the overlay network for its pods.
	// +kubebuilder:validation:Minimum=2
	// +kubebuilder:validation:Maximum=30
	HostSubnetLength uint32 `json:"hostSubnetLength" protobuf:"varint,2,opt,name=hostSubnetLength"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterNetworkList is a collection of ClusterNetworks
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ClusterNetworkList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of cluster networks
	Items []ClusterNetwork `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// HostSubnetEgressIP represents one egress IP address currently hosted on the node represented by
// HostSubnet
// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])$`
type HostSubnetEgressIP string

// HostSubnetEgressCIDR represents one egress CIDR from which to assign IP addresses for this node
// represented by the HostSubnet
// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])/([0-9]|[12][0-9]|3[0-2])$`
type HostSubnetEgressCIDR string

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// HostSubnet describes the container subnet network on a node. The HostSubnet object must have the
// same name as the Node object it corresponds to.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +kubebuilder:printcolumn:name="Host",type=string,JSONPath=`.host`,description="The name of the node"
// +kubebuilder:printcolumn:name="Host IP",type=string,JSONPath=`.hostIP`,description="The IP address to be used as a VTEP by other nodes in the overlay network"
// +kubebuilder:printcolumn:name="Subnet",type=string,JSONPath=`.subnet`,description="The CIDR range of the overlay network assigned to the node for its pods"
// +kubebuilder:printcolumn:name="Egress CIDRs",type=string,JSONPath=`.egressCIDRs`,description="The network egress CIDRs"
// +kubebuilder:printcolumn:name="Egress IPs",type=string,JSONPath=`.egressIPs`,description="The network egress IP addresses"
// +openshift:compatibility-gen:level=1
type HostSubnet struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Host is the name of the node. (This is the same as the object's name, but both fields must be set.)
	// +kubebuilder:validation:Pattern=`^[a-z0-9.-]+$`
	Host string `json:"host" protobuf:"bytes,2,opt,name=host"`

	// HostIP is the IP address to be used as a VTEP by other nodes in the overlay network
	// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])$`
	HostIP string `json:"hostIP" protobuf:"bytes,3,opt,name=hostIP"`

	// Subnet is the CIDR range of the overlay network assigned to the node for its pods
	// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])/([0-9]|[12][0-9]|3[0-2])$`
	Subnet string `json:"subnet" protobuf:"bytes,4,opt,name=subnet"`

	// EgressIPs is the list of automatic egress IP addresses currently hosted by this node.
	// If EgressCIDRs is empty, this can be set by hand; if EgressCIDRs is set then the
	// master will overwrite the value here with its own allocation of egress IPs.
	// +optional
	EgressIPs []HostSubnetEgressIP `json:"egressIPs,omitempty" protobuf:"bytes,5,rep,name=egressIPs"`

	// EgressCIDRs is the list of CIDR ranges available for automatically assigning
	// egress IPs to this node from. If this field is set then EgressIPs should be
	// treated as read-only.
	// +optional
	EgressCIDRs []HostSubnetEgressCIDR `json:"egressCIDRs,omitempty" protobuf:"bytes,6,rep,name=egressCIDRs"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// HostSubnetList is a collection of HostSubnets
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type HostSubnetList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of host subnets
	Items []HostSubnet `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// NetNamespaceEgressIP is a single egress IP out of a list of reserved IPs used as source of external traffic coming
// from pods in this namespace
// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])$`
type NetNamespaceEgressIP string

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NetNamespace describes a single isolated network. When using the redhat/openshift-ovs-multitenant
// plugin, every Namespace will have a corresponding NetNamespace object with the same name.
// (When using redhat/openshift-ovs-subnet, NetNamespaces are not used.)
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +kubebuilder:printcolumn:name="NetID",type=integer,JSONPath=`.netid`,description="The network identifier of the network namespace"
// +kubebuilder:printcolumn:name="Egress IPs",type=string,JSONPath=`.egressIPs`,description="The network egress IP addresses"
// +openshift:compatibility-gen:level=1
type NetNamespace struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// NetName is the name of the network namespace. (This is the same as the object's name, but both fields must be set.)
	// +kubebuilder:validation:Pattern=`^[a-z0-9.-]+$`
	NetName string `json:"netname" protobuf:"bytes,2,opt,name=netname"`

	// NetID is the network identifier of the network namespace assigned to each overlay network packet. This can be manipulated with the "oc adm pod-network" commands.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=16777215
	NetID uint32 `json:"netid" protobuf:"varint,3,opt,name=netid"`

	// EgressIPs is a list of reserved IPs that will be used as the source for external traffic coming from pods in this namespace.
	// (If empty, external traffic will be masqueraded to Node IPs.)
	// +optional
	EgressIPs []NetNamespaceEgressIP `json:"egressIPs,omitempty" protobuf:"bytes,4,rep,name=egressIPs"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NetNamespaceList is a collection of NetNamespaces
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type NetNamespaceList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of net namespaces
	Items []NetNamespace `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// EgressNetworkPolicyRuleType indicates whether an EgressNetworkPolicyRule allows or denies traffic
// +kubebuilder:validation:Pattern=`^Allow|Deny$`
type EgressNetworkPolicyRuleType string

const (
	EgressNetworkPolicyRuleAllow EgressNetworkPolicyRuleType = "Allow"
	EgressNetworkPolicyRuleDeny  EgressNetworkPolicyRuleType = "Deny"
)

// EgressNetworkPolicyPeer specifies a target to apply egress network policy to
type EgressNetworkPolicyPeer struct {
	// CIDRSelector is the CIDR range to allow/deny traffic to. If this is set, dnsName must be unset
	// Ideally we would have liked to use the cidr openapi format for this property.
	// But openshift-sdn only supports v4 while specifying the cidr format allows both v4 and v6 cidrs
	// We are therefore using a regex pattern to validate instead.
	// +kubebuilder:validation:Pattern=`^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])/([0-9]|[12][0-9]|3[0-2])$`
	CIDRSelector string `json:"cidrSelector,omitempty" protobuf:"bytes,1,rep,name=cidrSelector"`
	// DNSName is the domain name to allow/deny traffic to. If this is set, cidrSelector must be unset
	// +kubebuilder:validation:Pattern=`^([A-Za-z0-9-]+\.)*[A-Za-z0-9-]+\.?$`
	DNSName string `json:"dnsName,omitempty" protobuf:"bytes,2,rep,name=dnsName"`
}

// EgressNetworkPolicyRule contains a single egress network policy rule
type EgressNetworkPolicyRule struct {
	// type marks this as an "Allow" or "Deny" rule
	Type EgressNetworkPolicyRuleType `json:"type" protobuf:"bytes,1,rep,name=type"`
	// to is the target that traffic is allowed/denied to
	To EgressNetworkPolicyPeer `json:"to" protobuf:"bytes,2,rep,name=to"`
}

// EgressNetworkPolicySpec provides a list of policies on outgoing network traffic
type EgressNetworkPolicySpec struct {
	// egress contains the list of egress policy rules
	Egress []EgressNetworkPolicyRule `json:"egress" protobuf:"bytes,1,rep,name=egress"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EgressNetworkPolicy describes the current egress network policy for a Namespace. When using
// the 'redhat/openshift-ovs-multitenant' network plugin, traffic from a pod to an IP address
// outside the cluster will be checked against each EgressNetworkPolicyRule in the pod's
// namespace's EgressNetworkPolicy, in order. If no rule matches (or no EgressNetworkPolicy
// is present) then the traffic will be allowed by default.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type EgressNetworkPolicy struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec is the specification of the current egress network policy
	Spec EgressNetworkPolicySpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EgressNetworkPolicyList is a collection of EgressNetworkPolicy
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type EgressNetworkPolicyList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of policies
	Items []EgressNetworkPolicy `json:"items" protobuf:"bytes,2,rep,name=items"`
}
