package swarm

// Node represents a node.
type Node struct {
	ID string
	Meta

	Spec          NodeSpec        `json:",omitempty"`
	Description   NodeDescription `json:",omitempty"`
	Status        NodeStatus      `json:",omitempty"`
	ManagerStatus *ManagerStatus  `json:",omitempty"`
}

// NodeSpec represents the spec of a node.
type NodeSpec struct {
	Annotations
	Role         NodeRole         `json:",omitempty"`
	Membership   NodeMembership   `json:",omitempty"`
	Availability NodeAvailability `json:",omitempty"`
}

// NodeRole represents the role of a node.
type NodeRole string

const (
	// NodeRoleWorker WORKER
	NodeRoleWorker NodeRole = "worker"
	// NodeRoleManager MANAGER
	NodeRoleManager NodeRole = "manager"
)

// NodeMembership represents the membership of a node.
type NodeMembership string

const (
	// NodeMembershipPending PENDING
	NodeMembershipPending NodeMembership = "pending"
	// NodeMembershipAccepted ACCEPTED
	NodeMembershipAccepted NodeMembership = "accepted"
)

// NodeAvailability represents the availability of a node.
type NodeAvailability string

const (
	// NodeAvailabilityActive ACTIVE
	NodeAvailabilityActive NodeAvailability = "active"
	// NodeAvailabilityPause PAUSE
	NodeAvailabilityPause NodeAvailability = "pause"
	// NodeAvailabilityDrain DRAIN
	NodeAvailabilityDrain NodeAvailability = "drain"
)

// NodeDescription represents the description of a node.
type NodeDescription struct {
	Hostname  string            `json:",omitempty"`
	Platform  Platform          `json:",omitempty"`
	Resources Resources         `json:",omitempty"`
	Engine    EngineDescription `json:",omitempty"`
}

// Platform represents the platfrom (Arch/OS).
type Platform struct {
	Architecture string `json:",omitempty"`
	OS           string `json:",omitempty"`
}

// EngineDescription represents the description of an engine.
type EngineDescription struct {
	EngineVersion string              `json:",omitempty"`
	Labels        map[string]string   `json:",omitempty"`
	Plugins       []PluginDescription `json:",omitempty"`
}

// PluginDescription represents the description of an engine plugin.
type PluginDescription struct {
	Type string `json:",omitempty"`
	Name string `json:",omitempty"`
}

// NodeStatus represents the status of a node.
type NodeStatus struct {
	State   NodeState `json:",omitempty"`
	Message string    `json:",omitempty"`
}

// Reachability represents the reachability of a node.
type Reachability string

const (
	// ReachabilityUnknown UNKNOWN
	ReachabilityUnknown Reachability = "unknown"
	// ReachabilityUnreachable UNREACHABLE
	ReachabilityUnreachable Reachability = "unreachable"
	// ReachabilityReachable REACHABLE
	ReachabilityReachable Reachability = "reachable"
)

// ManagerStatus represents the status of a manager.
type ManagerStatus struct {
	Leader       bool         `json:",omitempty"`
	Reachability Reachability `json:",omitempty"`
	Addr         string       `json:",omitempty"`
}

// NodeState represents the state of a node.
type NodeState string

const (
	// NodeStateUnknown UNKNOWN
	NodeStateUnknown NodeState = "unknown"
	// NodeStateDown DOWN
	NodeStateDown NodeState = "down"
	// NodeStateReady READY
	NodeStateReady NodeState = "ready"
	// NodeStateDisconnected DISCONNECTED
	NodeStateDisconnected NodeState = "disconnected"
)
