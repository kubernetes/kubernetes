package server

import (
	"bytes"
	"io"

	"github.com/BurntSushi/toml"
	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

// Config provides containerd configuration data for the server
type Config struct {
	// Root is the path to a directory where containerd will store persistent data
	Root string `toml:"root"`
	// State is the path to a directory where containerd will store transient data
	State string `toml:"state"`
	// GRPC configuration settings
	GRPC GRPCConfig `toml:"grpc"`
	// Debug and profiling settings
	Debug Debug `toml:"debug"`
	// Metrics and monitoring settings
	Metrics MetricsConfig `toml:"metrics"`
	// Plugins provides plugin specific configuration for the initialization of a plugin
	Plugins map[string]toml.Primitive `toml:"plugins"`
	// Enable containerd as a subreaper
	Subreaper bool `toml:"subreaper"`
	// OOMScore adjust the containerd's oom score
	OOMScore int `toml:"oom_score"`
	// Cgroup specifies cgroup information for the containerd daemon process
	Cgroup CgroupConfig `toml:"cgroup"`

	md toml.MetaData
}

// GRPCConfig provides GRPC configuration for the socket
type GRPCConfig struct {
	Address string `toml:"address"`
	UID     int    `toml:"uid"`
	GID     int    `toml:"gid"`
}

// Debug provides debug configuration
type Debug struct {
	Address string `toml:"address"`
	UID     int    `toml:"uid"`
	GID     int    `toml:"gid"`
	Level   string `toml:"level"`
}

// MetricsConfig provides metrics configuration
type MetricsConfig struct {
	Address string `toml:"address"`
}

// CgroupConfig provides cgroup configuration
type CgroupConfig struct {
	Path string `toml:"path"`
}

// Decode unmarshals a plugin specific configuration by plugin id
func (c *Config) Decode(id string, v interface{}) (interface{}, error) {
	data, ok := c.Plugins[id]
	if !ok {
		return v, nil
	}
	if err := c.md.PrimitiveDecode(data, v); err != nil {
		return nil, err
	}
	return v, nil
}

// WriteTo marshals the config to the provided writer
func (c *Config) WriteTo(w io.Writer) (int64, error) {
	buf := bytes.NewBuffer(nil)
	e := toml.NewEncoder(buf)
	if err := e.Encode(c); err != nil {
		return 0, err
	}
	return io.Copy(w, buf)
}

// LoadConfig loads the containerd server config from the provided path
func LoadConfig(path string, v *Config) error {
	if v == nil {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "argument v must not be nil")
	}
	md, err := toml.DecodeFile(path, v)
	if err != nil {
		return err
	}
	v.md = md
	return nil
}
