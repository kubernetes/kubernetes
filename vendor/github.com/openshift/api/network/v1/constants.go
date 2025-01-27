package v1

const (
	// Pod annotations
	AssignMacvlanAnnotation = "pod.network.openshift.io/assign-macvlan"

	// HostSubnet annotations. (Note: should be "hostsubnet.network.openshift.io/", but the incorrect name is now part of the API.)
	AssignHostSubnetAnnotation = "pod.network.openshift.io/assign-subnet"
	FixedVNIDHostAnnotation    = "pod.network.openshift.io/fixed-vnid-host"
	NodeUIDAnnotation          = "pod.network.openshift.io/node-uid"

	// NetNamespace annotations
	MulticastEnabledAnnotation = "netnamespace.network.openshift.io/multicast-enabled"

	// ChangePodNetworkAnnotation is an annotation on NetNamespace to request change of pod network
	ChangePodNetworkAnnotation string = "pod.network.openshift.io/multitenant.change-network"
)
