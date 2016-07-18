package swarm

import "time"

// Service represents a service.
type Service struct {
	ID string
	Meta
	Spec     ServiceSpec `json:",omitempty"`
	Endpoint Endpoint    `json:",omitempty"`
}

// ServiceSpec represents the spec of a service.
type ServiceSpec struct {
	Annotations

	// TaskTemplate defines how the service should construct new tasks when
	// orchestrating this service.
	TaskTemplate TaskSpec                  `json:",omitempty"`
	Mode         ServiceMode               `json:",omitempty"`
	UpdateConfig *UpdateConfig             `json:",omitempty"`
	Networks     []NetworkAttachmentConfig `json:",omitempty"`
	EndpointSpec *EndpointSpec             `json:",omitempty"`
}

// ServiceMode represents the mode of a service.
type ServiceMode struct {
	Replicated *ReplicatedService `json:",omitempty"`
	Global     *GlobalService     `json:",omitempty"`
}

// ReplicatedService is a kind of ServiceMode.
type ReplicatedService struct {
	Replicas *uint64 `json:",omitempty"`
}

// GlobalService is a kind of ServiceMode.
type GlobalService struct{}

// UpdateConfig represents the update configuration.
type UpdateConfig struct {
	Parallelism uint64        `json:",omitempty"`
	Delay       time.Duration `json:",omitempty"`
}
