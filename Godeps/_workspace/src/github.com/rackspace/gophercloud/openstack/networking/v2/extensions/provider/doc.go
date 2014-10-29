// Package provider gives access to the provider Neutron plugin, allowing
// network extended attributes. The provider extended attributes for networks
// enable administrative users to specify how network objects map to the
// underlying networking infrastructure. These extended attributes also appear
// when administrative users query networks.
//
// For more information about extended attributes, see the NetworkExtAttrs
// struct. The actual semantics of these attributes depend on the technology
// back end of the particular plug-in. See the plug-in documentation and the
// OpenStack Cloud Administrator Guide to understand which values should be
// specific for each of these attributes when OpenStack Networking is deployed
// with a particular plug-in. The examples shown in this chapter refer to the
// Open vSwitch plug-in.
//
// The default policy settings enable only users with administrative rights to
// specify these parameters in requests and to see their values in responses. By
// default, the provider network extension attributes are completely hidden from
// regular tenants. As a rule of thumb, if these attributes are not visible in a
// GET /networks/<network-id> operation, this implies the user submitting the
// request is not authorized to view or manipulate provider network attributes.
package provider
