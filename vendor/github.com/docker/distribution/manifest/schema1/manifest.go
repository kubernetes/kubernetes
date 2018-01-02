package schema1

import (
	"encoding/json"
	"fmt"

	"github.com/docker/distribution"
	"github.com/docker/distribution/manifest"
	"github.com/docker/libtrust"
	"github.com/opencontainers/go-digest"
)

const (
	// MediaTypeManifest specifies the mediaType for the current version. Note
	// that for schema version 1, the the media is optionally "application/json".
	MediaTypeManifest = "application/vnd.docker.distribution.manifest.v1+json"
	// MediaTypeSignedManifest specifies the mediatype for current SignedManifest version
	MediaTypeSignedManifest = "application/vnd.docker.distribution.manifest.v1+prettyjws"
	// MediaTypeManifestLayer specifies the media type for manifest layers
	MediaTypeManifestLayer = "application/vnd.docker.container.image.rootfs.diff+x-gtar"
)

var (
	// SchemaVersion provides a pre-initialized version structure for this
	// packages version of the manifest.
	SchemaVersion = manifest.Versioned{
		SchemaVersion: 1,
	}
)

func init() {
	schema1Func := func(b []byte) (distribution.Manifest, distribution.Descriptor, error) {
		sm := new(SignedManifest)
		err := sm.UnmarshalJSON(b)
		if err != nil {
			return nil, distribution.Descriptor{}, err
		}

		desc := distribution.Descriptor{
			Digest:    digest.FromBytes(sm.Canonical),
			Size:      int64(len(sm.Canonical)),
			MediaType: MediaTypeSignedManifest,
		}
		return sm, desc, err
	}
	err := distribution.RegisterManifestSchema(MediaTypeSignedManifest, schema1Func)
	if err != nil {
		panic(fmt.Sprintf("Unable to register manifest: %s", err))
	}
	err = distribution.RegisterManifestSchema("", schema1Func)
	if err != nil {
		panic(fmt.Sprintf("Unable to register manifest: %s", err))
	}
	err = distribution.RegisterManifestSchema("application/json", schema1Func)
	if err != nil {
		panic(fmt.Sprintf("Unable to register manifest: %s", err))
	}
}

// FSLayer is a container struct for BlobSums defined in an image manifest
type FSLayer struct {
	// BlobSum is the tarsum of the referenced filesystem image layer
	BlobSum digest.Digest `json:"blobSum"`
}

// History stores unstructured v1 compatibility information
type History struct {
	// V1Compatibility is the raw v1 compatibility information
	V1Compatibility string `json:"v1Compatibility"`
}

// Manifest provides the base accessible fields for working with V2 image
// format in the registry.
type Manifest struct {
	manifest.Versioned

	// Name is the name of the image's repository
	Name string `json:"name"`

	// Tag is the tag of the image specified by this manifest
	Tag string `json:"tag"`

	// Architecture is the host architecture on which this image is intended to
	// run
	Architecture string `json:"architecture"`

	// FSLayers is a list of filesystem layer blobSums contained in this image
	FSLayers []FSLayer `json:"fsLayers"`

	// History is a list of unstructured historical data for v1 compatibility
	History []History `json:"history"`
}

// SignedManifest provides an envelope for a signed image manifest, including
// the format sensitive raw bytes.
type SignedManifest struct {
	Manifest

	// Canonical is the canonical byte representation of the ImageManifest,
	// without any attached signatures. The manifest byte
	// representation cannot change or it will have to be re-signed.
	Canonical []byte `json:"-"`

	// all contains the byte representation of the Manifest including signatures
	// and is returned by Payload()
	all []byte
}

// UnmarshalJSON populates a new SignedManifest struct from JSON data.
func (sm *SignedManifest) UnmarshalJSON(b []byte) error {
	sm.all = make([]byte, len(b), len(b))
	// store manifest and signatures in all
	copy(sm.all, b)

	jsig, err := libtrust.ParsePrettySignature(b, "signatures")
	if err != nil {
		return err
	}

	// Resolve the payload in the manifest.
	bytes, err := jsig.Payload()
	if err != nil {
		return err
	}

	// sm.Canonical stores the canonical manifest JSON
	sm.Canonical = make([]byte, len(bytes), len(bytes))
	copy(sm.Canonical, bytes)

	// Unmarshal canonical JSON into Manifest object
	var manifest Manifest
	if err := json.Unmarshal(sm.Canonical, &manifest); err != nil {
		return err
	}

	sm.Manifest = manifest

	return nil
}

// References returnes the descriptors of this manifests references
func (sm SignedManifest) References() []distribution.Descriptor {
	dependencies := make([]distribution.Descriptor, len(sm.FSLayers))
	for i, fsLayer := range sm.FSLayers {
		dependencies[i] = distribution.Descriptor{
			MediaType: "application/vnd.docker.container.image.rootfs.diff+x-gtar",
			Digest:    fsLayer.BlobSum,
		}
	}

	return dependencies

}

// MarshalJSON returns the contents of raw. If Raw is nil, marshals the inner
// contents. Applications requiring a marshaled signed manifest should simply
// use Raw directly, since the the content produced by json.Marshal will be
// compacted and will fail signature checks.
func (sm *SignedManifest) MarshalJSON() ([]byte, error) {
	if len(sm.all) > 0 {
		return sm.all, nil
	}

	// If the raw data is not available, just dump the inner content.
	return json.Marshal(&sm.Manifest)
}

// Payload returns the signed content of the signed manifest.
func (sm SignedManifest) Payload() (string, []byte, error) {
	return MediaTypeSignedManifest, sm.all, nil
}

// Signatures returns the signatures as provided by
// (*libtrust.JSONSignature).Signatures. The byte slices are opaque jws
// signatures.
func (sm *SignedManifest) Signatures() ([][]byte, error) {
	jsig, err := libtrust.ParsePrettySignature(sm.all, "signatures")
	if err != nil {
		return nil, err
	}

	// Resolve the payload in the manifest.
	return jsig.Signatures()
}
