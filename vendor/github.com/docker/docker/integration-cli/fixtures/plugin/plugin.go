package plugin

import (
	"io"

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

// CreateOpt is is passed used to change the defualt plugin config before
// creating it
type CreateOpt func(*Config)

// Config wraps types.PluginConfig to provide some extra state for options
// extra customizations on the plugin details, such as using a custom binary to
// create the plugin with.
type Config struct {
	*types.PluginConfig
	binPath string
}

// WithBinary is a CreateOpt to set an custom binary to create the plugin with.
// This binary must be statically compiled.
func WithBinary(bin string) CreateOpt {
	return func(cfg *Config) {
		cfg.binPath = bin
	}
}

// CreateClient is the interface used for `BuildPlugin` to interact with the
// daemon.
type CreateClient interface {
	PluginCreate(context.Context, io.Reader, types.PluginCreateOptions) error
}
