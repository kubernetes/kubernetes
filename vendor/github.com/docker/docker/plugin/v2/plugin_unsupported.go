// +build !linux

package v2

import (
	"errors"

	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// InitSpec creates an OCI spec from the plugin's config.
func (p *Plugin) InitSpec(execRoot string) (*specs.Spec, error) {
	return nil, errors.New("not supported")
}
