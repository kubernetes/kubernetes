package swarm

import "time"

// Version represents the internal object version.
type Version struct {
	Index uint64 `json:",omitempty"`
}

// Meta is a base object inherited by most of the other once.
type Meta struct {
	Version   Version   `json:",omitempty"`
	CreatedAt time.Time `json:",omitempty"`
	UpdatedAt time.Time `json:",omitempty"`
}

// Annotations represents how to describe an object.
type Annotations struct {
	Name   string            `json:",omitempty"`
	Labels map[string]string `json:",omitempty"`
}

// Driver represents a driver (network, logging).
type Driver struct {
	Name    string            `json:",omitempty"`
	Options map[string]string `json:",omitempty"`
}
