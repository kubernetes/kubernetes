package schema2

import (
	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/opencontainers/go-digest"
)

// builder is a type for constructing manifests.
type builder struct {
	// bs is a BlobService used to publish the configuration blob.
	bs distribution.BlobService

	// configMediaType is media type used to describe configuration
	configMediaType string

	// configJSON references
	configJSON []byte

	// dependencies is a list of descriptors that gets built by successive
	// calls to AppendReference. In case of image configuration these are layers.
	dependencies []distribution.Descriptor
}

// NewManifestBuilder is used to build new manifests for the current schema
// version. It takes a BlobService so it can publish the configuration blob
// as part of the Build process.
func NewManifestBuilder(bs distribution.BlobService, configMediaType string, configJSON []byte) distribution.ManifestBuilder {
	mb := &builder{
		bs:              bs,
		configMediaType: configMediaType,
		configJSON:      make([]byte, len(configJSON)),
	}
	copy(mb.configJSON, configJSON)

	return mb
}

// Build produces a final manifest from the given references.
func (mb *builder) Build(ctx context.Context) (distribution.Manifest, error) {
	m := Manifest{
		Versioned: SchemaVersion,
		Layers:    make([]distribution.Descriptor, len(mb.dependencies)),
	}
	copy(m.Layers, mb.dependencies)

	configDigest := digest.FromBytes(mb.configJSON)

	var err error
	m.Config, err = mb.bs.Stat(ctx, configDigest)
	switch err {
	case nil:
		// Override MediaType, since Put always replaces the specified media
		// type with application/octet-stream in the descriptor it returns.
		m.Config.MediaType = mb.configMediaType
		return FromStruct(m)
	case distribution.ErrBlobUnknown:
		// nop
	default:
		return nil, err
	}

	// Add config to the blob store
	m.Config, err = mb.bs.Put(ctx, mb.configMediaType, mb.configJSON)
	// Override MediaType, since Put always replaces the specified media
	// type with application/octet-stream in the descriptor it returns.
	m.Config.MediaType = mb.configMediaType
	if err != nil {
		return nil, err
	}

	return FromStruct(m)
}

// AppendReference adds a reference to the current ManifestBuilder.
func (mb *builder) AppendReference(d distribution.Describable) error {
	mb.dependencies = append(mb.dependencies, d.Descriptor())
	return nil
}

// References returns the current references added to this builder.
func (mb *builder) References() []distribution.Descriptor {
	return mb.dependencies
}
