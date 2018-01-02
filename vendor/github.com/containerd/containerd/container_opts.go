package containerd

import (
	"context"

	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/platforms"
	"github.com/containerd/typeurl"
	"github.com/gogo/protobuf/types"
	"github.com/opencontainers/image-spec/identity"
	"github.com/pkg/errors"
)

// DeleteOpts allows the caller to set options for the deletion of a container
type DeleteOpts func(ctx context.Context, client *Client, c containers.Container) error

// NewContainerOpts allows the caller to set additional options when creating a container
type NewContainerOpts func(ctx context.Context, client *Client, c *containers.Container) error

// UpdateContainerOpts allows the caller to set additional options when updating a container
type UpdateContainerOpts func(ctx context.Context, client *Client, c *containers.Container) error

// WithRuntime allows a user to specify the runtime name and additional options that should
// be used to create tasks for the container
func WithRuntime(name string, options interface{}) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		var (
			any *types.Any
			err error
		)
		if options != nil {
			any, err = typeurl.MarshalAny(options)
			if err != nil {
				return err
			}
		}
		c.Runtime = containers.RuntimeInfo{
			Name:    name,
			Options: any,
		}
		return nil
	}
}

// WithImage sets the provided image as the base for the container
func WithImage(i Image) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		c.Image = i.Name()
		return nil
	}
}

// WithContainerLabels adds the provided labels to the container
func WithContainerLabels(labels map[string]string) NewContainerOpts {
	return func(_ context.Context, _ *Client, c *containers.Container) error {
		c.Labels = labels
		return nil
	}
}

// WithSnapshotter sets the provided snapshotter for use by the container
//
// This option must appear before other snapshotter options to have an effect.
func WithSnapshotter(name string) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		c.Snapshotter = name
		return nil
	}
}

// WithSnapshot uses an existing root filesystem for the container
func WithSnapshot(id string) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		setSnapshotterIfEmpty(c)
		// check that the snapshot exists, if not, fail on creation
		if _, err := client.SnapshotService(c.Snapshotter).Mounts(ctx, id); err != nil {
			return err
		}
		c.SnapshotKey = id
		return nil
	}
}

// WithNewSnapshot allocates a new snapshot to be used by the container as the
// root filesystem in read-write mode
func WithNewSnapshot(id string, i Image) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		diffIDs, err := i.(*image).i.RootFS(ctx, client.ContentStore(), platforms.Default())
		if err != nil {
			return err
		}
		setSnapshotterIfEmpty(c)
		parent := identity.ChainID(diffIDs).String()
		if _, err := client.SnapshotService(c.Snapshotter).Prepare(ctx, id, parent); err != nil {
			return err
		}
		c.SnapshotKey = id
		c.Image = i.Name()
		return nil
	}
}

// WithSnapshotCleanup deletes the rootfs snapshot allocated for the container
func WithSnapshotCleanup(ctx context.Context, client *Client, c containers.Container) error {
	if c.SnapshotKey != "" {
		if c.Snapshotter == "" {
			return errors.Wrapf(errdefs.ErrInvalidArgument, "container.Snapshotter must be set to cleanup rootfs snapshot")
		}
		return client.SnapshotService(c.Snapshotter).Remove(ctx, c.SnapshotKey)
	}
	return nil
}

// WithNewSnapshotView allocates a new snapshot to be used by the container as the
// root filesystem in read-only mode
func WithNewSnapshotView(id string, i Image) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		diffIDs, err := i.(*image).i.RootFS(ctx, client.ContentStore(), platforms.Default())
		if err != nil {
			return err
		}
		setSnapshotterIfEmpty(c)
		parent := identity.ChainID(diffIDs).String()
		if _, err := client.SnapshotService(c.Snapshotter).View(ctx, id, parent); err != nil {
			return err
		}
		c.SnapshotKey = id
		c.Image = i.Name()
		return nil
	}
}

func setSnapshotterIfEmpty(c *containers.Container) {
	if c.Snapshotter == "" {
		c.Snapshotter = DefaultSnapshotter
	}
}

// WithContainerExtension appends extension data to the container object.
// Use this to decorate the container object with additional data for the client
// integration.
//
// Make sure to register the type of `extension` in the typeurl package via
// `typeurl.Register` or container creation may fail.
func WithContainerExtension(name string, extension interface{}) NewContainerOpts {
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		if name == "" {
			return errors.Wrapf(errdefs.ErrInvalidArgument, "extension key must not be zero-length")
		}

		any, err := typeurl.MarshalAny(extension)
		if err != nil {
			if errors.Cause(err) == typeurl.ErrNotFound {
				return errors.Wrapf(err, "extension %q is not registered with the typeurl package, see `typeurl.Register`", name)
			}
			return errors.Wrap(err, "error marshalling extension")
		}

		if c.Extensions == nil {
			c.Extensions = make(map[string]types.Any)
		}
		c.Extensions[name] = *any
		return nil
	}
}
