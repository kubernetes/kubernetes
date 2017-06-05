package distribution

import (
	"fmt"
	"mime"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
)

// Manifest represents a registry object specifying a set of
// references and an optional target
type Manifest interface {
	// References returns a list of objects which make up this manifest.
	// The references are strictly ordered from base to head. A reference
	// is anything which can be represented by a distribution.Descriptor
	References() []Descriptor

	// Payload provides the serialized format of the manifest, in addition to
	// the mediatype.
	Payload() (mediatype string, payload []byte, err error)
}

// ManifestBuilder creates a manifest allowing one to include dependencies.
// Instances can be obtained from a version-specific manifest package.  Manifest
// specific data is passed into the function which creates the builder.
type ManifestBuilder interface {
	// Build creates the manifest from his builder.
	Build(ctx context.Context) (Manifest, error)

	// References returns a list of objects which have been added to this
	// builder. The dependencies are returned in the order they were added,
	// which should be from base to head.
	References() []Descriptor

	// AppendReference includes the given object in the manifest after any
	// existing dependencies. If the add fails, such as when adding an
	// unsupported dependency, an error may be returned.
	AppendReference(dependency Describable) error
}

// ManifestService describes operations on image manifests.
type ManifestService interface {
	// Exists returns true if the manifest exists.
	Exists(ctx context.Context, dgst digest.Digest) (bool, error)

	// Get retrieves the manifest specified by the given digest
	Get(ctx context.Context, dgst digest.Digest, options ...ManifestServiceOption) (Manifest, error)

	// Put creates or updates the given manifest returning the manifest digest
	Put(ctx context.Context, manifest Manifest, options ...ManifestServiceOption) (digest.Digest, error)

	// Delete removes the manifest specified by the given digest. Deleting
	// a manifest that doesn't exist will return ErrManifestNotFound
	Delete(ctx context.Context, dgst digest.Digest) error
}

// ManifestEnumerator enables iterating over manifests
type ManifestEnumerator interface {
	// Enumerate calls ingester for each manifest.
	Enumerate(ctx context.Context, ingester func(digest.Digest) error) error
}

// SignaturesGetter provides an interface for getting the signatures of a schema1 manifest. If the digest
// referred to is not a schema1 manifest, an error should be returned.
type SignaturesGetter interface {
	GetSignatures(ctx context.Context, manifestDigest digest.Digest) ([]digest.Digest, error)
}

// Describable is an interface for descriptors
type Describable interface {
	Descriptor() Descriptor
}

// ManifestMediaTypes returns the supported media types for manifests.
func ManifestMediaTypes() (mediaTypes []string) {
	for t := range mappings {
		if t != "" {
			mediaTypes = append(mediaTypes, t)
		}
	}
	return
}

// UnmarshalFunc implements manifest unmarshalling a given MediaType
type UnmarshalFunc func([]byte) (Manifest, Descriptor, error)

var mappings = make(map[string]UnmarshalFunc, 0)

// UnmarshalManifest looks up manifest unmarshal functions based on
// MediaType
func UnmarshalManifest(ctHeader string, p []byte) (Manifest, Descriptor, error) {
	// Need to look up by the actual media type, not the raw contents of
	// the header. Strip semicolons and anything following them.
	var mediatype string
	if ctHeader != "" {
		var err error
		mediatype, _, err = mime.ParseMediaType(ctHeader)
		if err != nil {
			return nil, Descriptor{}, err
		}
	}

	unmarshalFunc, ok := mappings[mediatype]
	if !ok {
		unmarshalFunc, ok = mappings[""]
		if !ok {
			return nil, Descriptor{}, fmt.Errorf("unsupported manifest mediatype and no default available: %s", mediatype)
		}
	}

	return unmarshalFunc(p)
}

// RegisterManifestSchema registers an UnmarshalFunc for a given schema type.  This
// should be called from specific
func RegisterManifestSchema(mediatype string, u UnmarshalFunc) error {
	if _, ok := mappings[mediatype]; ok {
		return fmt.Errorf("manifest mediatype registration would overwrite existing: %s", mediatype)
	}
	mappings[mediatype] = u
	return nil
}
