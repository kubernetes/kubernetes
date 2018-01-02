package remotes

import (
	"context"
	"io"

	"github.com/containerd/containerd/content"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// Resolver provides remotes based on a locator.
type Resolver interface {
	// Resolve attempts to resolve the reference into a name and descriptor.
	//
	// The argument `ref` should be a scheme-less URI representing the remote.
	// Structurally, it has a host and path. The "host" can be used to directly
	// reference a specific host or be matched against a specific handler.
	//
	// The returned name should be used to identify the referenced entity.
	// Dependending on the remote namespace, this may be immutable or mutable.
	// While the name may differ from ref, it should itself be a valid ref.
	//
	// If the resolution fails, an error will be returned.
	Resolve(ctx context.Context, ref string) (name string, desc ocispec.Descriptor, err error)

	// Fetcher returns a new fetcher for the provided reference.
	// All content fetched from the returned fetcher will be
	// from the namespace referred to by ref.
	Fetcher(ctx context.Context, ref string) (Fetcher, error)

	// Pusher returns a new pusher for the provided reference
	Pusher(ctx context.Context, ref string) (Pusher, error)
}

// Fetcher fetches content
type Fetcher interface {
	// Fetch the resource identified by the descriptor.
	Fetch(ctx context.Context, desc ocispec.Descriptor) (io.ReadCloser, error)
}

// Pusher pushes content
type Pusher interface {
	// Push returns a content writer for the given resource identified
	// by the descriptor.
	Push(ctx context.Context, d ocispec.Descriptor) (content.Writer, error)
}

// FetcherFunc allows package users to implement a Fetcher with just a
// function.
type FetcherFunc func(ctx context.Context, desc ocispec.Descriptor) (io.ReadCloser, error)

// Fetch content
func (fn FetcherFunc) Fetch(ctx context.Context, desc ocispec.Descriptor) (io.ReadCloser, error) {
	return fn(ctx, desc)
}

// PusherFunc allows package users to implement a Pusher with just a
// function.
type PusherFunc func(ctx context.Context, desc ocispec.Descriptor, r io.Reader) error

// Push content
func (fn PusherFunc) Push(ctx context.Context, desc ocispec.Descriptor, r io.Reader) error {
	return fn(ctx, desc, r)
}
