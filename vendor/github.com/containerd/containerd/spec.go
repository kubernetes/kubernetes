package containerd

import (
	"context"

	"github.com/containerd/containerd/containers"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// GenerateSpec will generate a default spec from the provided image
// for use as a containerd container
func GenerateSpec(ctx context.Context, client *Client, c *containers.Container, opts ...SpecOpts) (*specs.Spec, error) {
	s, err := createDefaultSpec(ctx, c.ID)
	if err != nil {
		return nil, err
	}
	for _, o := range opts {
		if err := o(ctx, client, c, s); err != nil {
			return nil, err
		}
	}
	return s, nil
}
