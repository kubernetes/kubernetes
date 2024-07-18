package swarm // import "github.com/docker/docker/api/types/swarm"

import "os"

// Secret represents a secret.
type Secret struct {
	ID string
	Meta
	Spec SecretSpec
}

// SecretSpec represents a secret specification from a secret in swarm
type SecretSpec struct {
	Annotations
	Data   []byte  `json:",omitempty"`
	Driver *Driver `json:",omitempty"` // name of the secrets driver used to fetch the secret's value from an external secret store

	// Templating controls whether and how to evaluate the secret payload as
	// a template. If it is not set, no templating is used.
	Templating *Driver `json:",omitempty"`
}

// SecretReferenceFileTarget is a file target in a secret reference
type SecretReferenceFileTarget struct {
	Name string
	UID  string
	GID  string
	Mode os.FileMode
}

// SecretReference is a reference to a secret in swarm
type SecretReference struct {
	File       *SecretReferenceFileTarget
	SecretID   string
	SecretName string
}
