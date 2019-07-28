package swarm // import "github.com/docker/docker/api/types/swarm"

import "os"

// Config represents a config.
type Config struct {
	ID string
	Meta
	Spec ConfigSpec
}

// ConfigSpec represents a config specification from a config in swarm
type ConfigSpec struct {
	Annotations
	Data []byte `json:",omitempty"`

	// Templating controls whether and how to evaluate the config payload as
	// a template. If it is not set, no templating is used.
	Templating *Driver `json:",omitempty"`
}

// ConfigReferenceFileTarget is a file target in a config reference
type ConfigReferenceFileTarget struct {
	Name string
	UID  string
	GID  string
	Mode os.FileMode
}

// ConfigReferenceRuntimeTarget is a target for a config specifying that it
// isn't mounted into the container but instead has some other purpose.
type ConfigReferenceRuntimeTarget struct{}

// ConfigReference is a reference to a config in swarm
type ConfigReference struct {
	File       *ConfigReferenceFileTarget    `json:",omitempty"`
	Runtime    *ConfigReferenceRuntimeTarget `json:",omitempty"`
	ConfigID   string
	ConfigName string
}
