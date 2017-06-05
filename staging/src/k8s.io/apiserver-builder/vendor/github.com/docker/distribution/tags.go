package distribution

import (
	"github.com/docker/distribution/context"
)

// TagService provides access to information about tagged objects.
type TagService interface {
	// Get retrieves the descriptor identified by the tag. Some
	// implementations may differentiate between "trusted" tags and
	// "untrusted" tags. If a tag is "untrusted", the mapping will be returned
	// as an ErrTagUntrusted error, with the target descriptor.
	Get(ctx context.Context, tag string) (Descriptor, error)

	// Tag associates the tag with the provided descriptor, updating the
	// current association, if needed.
	Tag(ctx context.Context, tag string, desc Descriptor) error

	// Untag removes the given tag association
	Untag(ctx context.Context, tag string) error

	// All returns the set of tags managed by this tag service
	All(ctx context.Context) ([]string, error)

	// Lookup returns the set of tags referencing the given digest.
	Lookup(ctx context.Context, digest Descriptor) ([]string, error)
}
