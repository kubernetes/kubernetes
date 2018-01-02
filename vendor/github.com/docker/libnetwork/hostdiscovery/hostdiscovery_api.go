package hostdiscovery

import "net"

// JoinCallback provides a callback event for new node joining the cluster
type JoinCallback func(entries []net.IP)

// ActiveCallback provides a callback event for active discovery event
type ActiveCallback func()

// LeaveCallback provides a callback event for node leaving the cluster
type LeaveCallback func(entries []net.IP)

// HostDiscovery primary interface
type HostDiscovery interface {
	//Watch Node join and leave cluster events
	Watch(activeCallback ActiveCallback, joinCallback JoinCallback, leaveCallback LeaveCallback) error
	// StopDiscovery stops the discovery process
	StopDiscovery() error
	// Fetch returns a list of host IPs that are currently discovered
	Fetch() []net.IP
}
