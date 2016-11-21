// Package networks contains functionality for working with Neutron network
// resources. A network is an isolated virtual layer-2 broadcast domain that is
// typically reserved for the tenant who created it (unless you configure the
// network to be shared). Tenants can create multiple networks until the
// thresholds per-tenant quota is reached.
//
// In the v2.0 Networking API, the network is the main entity. Ports and subnets
// are always associated with a network.
package networks
