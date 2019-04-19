package events // import "github.com/docker/docker/api/types/events"

const (
	// ContainerEventType is the event type that containers generate
	ContainerEventType = "container"
	// DaemonEventType is the event type that daemon generate
	DaemonEventType = "daemon"
	// ImageEventType is the event type that images generate
	ImageEventType = "image"
	// NetworkEventType is the event type that networks generate
	NetworkEventType = "network"
	// PluginEventType is the event type that plugins generate
	PluginEventType = "plugin"
	// VolumeEventType is the event type that volumes generate
	VolumeEventType = "volume"
	// ServiceEventType is the event type that services generate
	ServiceEventType = "service"
	// NodeEventType is the event type that nodes generate
	NodeEventType = "node"
	// SecretEventType is the event type that secrets generate
	SecretEventType = "secret"
	// ConfigEventType is the event type that configs generate
	ConfigEventType = "config"
)

// Actor describes something that generates events,
// like a container, or a network, or a volume.
// It has a defined name and a set or attributes.
// The container attributes are its labels, other actors
// can generate these attributes from other properties.
type Actor struct {
	ID         string
	Attributes map[string]string
}

// Message represents the information an event contains
type Message struct {
	// Deprecated information from JSONMessage.
	// With data only in container events.
	Status string `json:"status,omitempty"`
	ID     string `json:"id,omitempty"`
	From   string `json:"from,omitempty"`

	Type   string
	Action string
	Actor  Actor
	// Engine events are local scope. Cluster events are swarm scope.
	Scope string `json:"scope,omitempty"`

	Time     int64 `json:"time,omitempty"`
	TimeNano int64 `json:"timeNano,omitempty"`
}
