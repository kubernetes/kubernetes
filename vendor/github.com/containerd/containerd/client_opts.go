package containerd

import (
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/remotes"
	"google.golang.org/grpc"
)

type clientOpts struct {
	defaultns   string
	dialOptions []grpc.DialOption
}

// ClientOpt allows callers to set options on the containerd client
type ClientOpt func(c *clientOpts) error

// WithDefaultNamespace sets the default namespace on the client
//
// Any operation that does not have a namespace set on the context will
// be provided the default namespace
func WithDefaultNamespace(ns string) ClientOpt {
	return func(c *clientOpts) error {
		c.defaultns = ns
		return nil
	}
}

// WithDialOpts allows grpc.DialOptions to be set on the connection
func WithDialOpts(opts []grpc.DialOption) ClientOpt {
	return func(c *clientOpts) error {
		c.dialOptions = opts
		return nil
	}
}

// RemoteOpt allows the caller to set distribution options for a remote
type RemoteOpt func(*Client, *RemoteContext) error

// WithPullUnpack is used to unpack an image after pull. This
// uses the snapshotter, content store, and diff service
// configured for the client.
func WithPullUnpack(_ *Client, c *RemoteContext) error {
	c.Unpack = true
	return nil
}

// WithPullSnapshotter specifies snapshotter name used for unpacking
func WithPullSnapshotter(snapshotterName string) RemoteOpt {
	return func(_ *Client, c *RemoteContext) error {
		c.Snapshotter = snapshotterName
		return nil
	}
}

// WithPullLabel sets a label to be associated with a pulled reference
func WithPullLabel(key, value string) RemoteOpt {
	return func(_ *Client, rc *RemoteContext) error {
		if rc.Labels == nil {
			rc.Labels = make(map[string]string)
		}

		rc.Labels[key] = value
		return nil
	}
}

// WithPullLabels associates a set of labels to a pulled reference
func WithPullLabels(labels map[string]string) RemoteOpt {
	return func(_ *Client, rc *RemoteContext) error {
		if rc.Labels == nil {
			rc.Labels = make(map[string]string)
		}

		for k, v := range labels {
			rc.Labels[k] = v
		}
		return nil
	}
}

// WithSchema1Conversion is used to convert Docker registry schema 1
// manifests to oci manifests on pull. Without this option schema 1
// manifests will return a not supported error.
func WithSchema1Conversion(client *Client, c *RemoteContext) error {
	c.ConvertSchema1 = true
	return nil
}

// WithResolver specifies the resolver to use.
func WithResolver(resolver remotes.Resolver) RemoteOpt {
	return func(client *Client, c *RemoteContext) error {
		c.Resolver = resolver
		return nil
	}
}

// WithImageHandler adds a base handler to be called on dispatch.
func WithImageHandler(h images.Handler) RemoteOpt {
	return func(client *Client, c *RemoteContext) error {
		c.BaseHandlers = append(c.BaseHandlers, h)
		return nil
	}
}
