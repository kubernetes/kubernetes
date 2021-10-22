// Package security contains functionality to work with security group and
// security group rules Neutron resources.
//
// Security groups and security group rules allows administrators and tenants
// the ability to specify the type of traffic and direction (ingress/egress)
// that is allowed to pass through a port. A security group is a container for
// security group rules.
//
// When a port is created in Networking it is associated with a security group.
// If a security group is not specified the port is associated with a 'default'
// security group. By default, this group drops all ingress traffic and allows
// all egress. Rules can be added to this group in order to change the behaviour.
//
// The basic characteristics of Neutron Security Groups are:
//
// For ingress traffic (to an instance)
//  - Only traffic matched with security group rules are allowed.
//  - When there is no rule defined, all traffic is dropped.
//
// For egress traffic (from an instance)
//  - Only traffic matched with security group rules are allowed.
//  - When there is no rule defined, all egress traffic are dropped.
//  - When a new security group is created, rules to allow all egress traffic
//    is automatically added.
//
// "default security group" is defined for each tenant.
//  - For the default security group a rule which allows intercommunication
//    among hosts associated with the default security group is defined by default.
//  - As a result, all egress traffic and intercommunication in the default
//    group are allowed and all ingress from outside of the default group is
//    dropped by default (in the default security group).
package security
