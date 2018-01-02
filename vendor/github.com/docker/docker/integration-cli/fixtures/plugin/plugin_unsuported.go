// +build !linux

package plugin

import (
	"github.com/docker/docker/api/types"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

// Create is not supported on this platform
func Create(ctx context.Context, c CreateClient, name string, opts ...CreateOpt) error {
	return errors.New("not supported on this platform")
}

// CreateInRegistry is not supported on this platform
func CreateInRegistry(ctx context.Context, repo string, auth *types.AuthConfig, opts ...CreateOpt) error {
	return errors.New("not supported on this platform")
}
