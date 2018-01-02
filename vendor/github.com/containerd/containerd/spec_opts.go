package containerd

import (
	"context"

	"github.com/containerd/containerd/containers"
	"github.com/containerd/typeurl"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// SpecOpts sets spec specific information to a newly generated OCI spec
type SpecOpts func(context.Context, *Client, *containers.Container, *specs.Spec) error

// WithProcessArgs replaces the args on the generated spec
func WithProcessArgs(args ...string) SpecOpts {
	return func(_ context.Context, _ *Client, _ *containers.Container, s *specs.Spec) error {
		s.Process.Args = args
		return nil
	}
}

// WithProcessCwd replaces the current working directory on the generated spec
func WithProcessCwd(cwd string) SpecOpts {
	return func(_ context.Context, _ *Client, _ *containers.Container, s *specs.Spec) error {
		s.Process.Cwd = cwd
		return nil
	}
}

// WithHostname sets the container's hostname
func WithHostname(name string) SpecOpts {
	return func(_ context.Context, _ *Client, _ *containers.Container, s *specs.Spec) error {
		s.Hostname = name
		return nil
	}
}

// WithNewSpec generates a new spec for a new container
func WithNewSpec(opts ...SpecOpts) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		s, err := createDefaultSpec(ctx, c.ID)
		if err != nil {
			return err
		}
		for _, o := range opts {
			if err := o(ctx, client, c, s); err != nil {
				return err
			}
		}
		any, err := typeurl.MarshalAny(s)
		if err != nil {
			return err
		}
		c.Spec = any
		return nil
	}
}

// WithSpec sets the provided spec on the container
func WithSpec(s *specs.Spec, opts ...SpecOpts) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		for _, o := range opts {
			if err := o(ctx, client, c, s); err != nil {
				return err
			}
		}
		any, err := typeurl.MarshalAny(s)
		if err != nil {
			return err
		}
		c.Spec = any
		return nil
	}
}
