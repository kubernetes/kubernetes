package schema2

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/docker/distribution"
	"github.com/docker/distribution/manifest"
	"github.com/opencontainers/go-digest"
)

const (
	// MediaTypeManifest specifies the mediaType for the current version.
	MediaTypeManifest = "application/vnd.docker.distribution.manifest.v2+json"

	// MediaTypeImageConfig specifies the mediaType for the image configuration.
	MediaTypeImageConfig = "application/vnd.docker.container.image.v1+json"

	// MediaTypePluginConfig specifies the mediaType for plugin configuration.
	MediaTypePluginConfig = "application/vnd.docker.plugin.v1+json"

	// MediaTypeLayer is the mediaType used for layers referenced by the
	// manifest.
	MediaTypeLayer = "application/vnd.docker.image.rootfs.diff.tar.gzip"

	// MediaTypeForeignLayer is the mediaType used for layers that must be
	// downloaded from foreign URLs.
	MediaTypeForeignLayer = "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip"

	// MediaTypeUncompressedLayer is the mediaType used for layers which
	// are not compressed.
	MediaTypeUncompressedLayer = "application/vnd.docker.image.rootfs.diff.tar"
)

var (
	// SchemaVersion provides a pre-initialized version structure for this
	// packages version of the manifest.
	SchemaVersion = manifest.Versioned{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
	}
)

func init() {
	schema2Func := func(b []byte) (distribution.Manifest, distribution.Descriptor, error) {
		m := new(DeserializedManifest)
		err := m.UnmarshalJSON(b)
		if err != nil {
			return nil, distribution.Descriptor{}, err
		}

		dgst := digest.FromBytes(b)
		return m, distribution.Descriptor{Digest: dgst, Size: int64(len(b)), MediaType: MediaTypeManifest}, err
	}
	err := distribution.RegisterManifestSchema(MediaTypeManifest, schema2Func)
	if err != nil {
		panic(fmt.Sprintf("Unable to register manifest: %s", err))
	}
}

// Manifest defines a schema2 manifest.
type Manifest struct {
	manifest.Versioned

	// Config references the image configuration as a blob.
	Config distribution.Descriptor `json:"config"`

	// Layers lists descriptors for the layers referenced by the
	// configuration.
	Layers []distribution.Descriptor `json:"layers"`
}

// References returnes the descriptors of this manifests references.
func (m Manifest) References() []distribution.Descriptor {
	references := make([]distribution.Descriptor, 0, 1+len(m.Layers))
	references = append(references, m.Config)
	references = append(references, m.Layers...)
	return references
}

// Target returns the target of this signed manifest.
func (m Manifest) Target() distribution.Descriptor {
	return m.Config
}

// DeserializedManifest wraps Manifest with a copy of the original JSON.
// It satisfies the distribution.Manifest interface.
type DeserializedManifest struct {
	Manifest

	// canonical is the canonical byte representation of the Manifest.
	canonical []byte
}

// FromStruct takes a Manifest structure, marshals it to JSON, and returns a
// DeserializedManifest which contains the manifest and its JSON representation.
func FromStruct(m Manifest) (*DeserializedManifest, error) {
	var deserialized DeserializedManifest
	deserialized.Manifest = m

	var err error
	deserialized.canonical, err = json.MarshalIndent(&m, "", "   ")
	return &deserialized, err
}

// UnmarshalJSON populates a new Manifest struct from JSON data.
func (m *DeserializedManifest) UnmarshalJSON(b []byte) error {
	m.canonical = make([]byte, len(b), len(b))
	// store manifest in canonical
	copy(m.canonical, b)

	// Unmarshal canonical JSON into Manifest object
	var manifest Manifest
	if err := json.Unmarshal(m.canonical, &manifest); err != nil {
		return err
	}

	m.Manifest = manifest

	return nil
}

// MarshalJSON returns the contents of canonical. If canonical is empty,
// marshals the inner contents.
func (m *DeserializedManifest) MarshalJSON() ([]byte, error) {
	if len(m.canonical) > 0 {
		return m.canonical, nil
	}

	return nil, errors.New("JSON representation not initialized in DeserializedManifest")
}

// Payload returns the raw content of the manifest. The contents can be used to
// calculate the content identifier.
func (m DeserializedManifest) Payload() (string, []byte, error) {
	return m.MediaType, m.canonical, nil
}
