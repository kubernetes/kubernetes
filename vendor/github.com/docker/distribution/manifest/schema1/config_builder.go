package schema1

import (
	"crypto/sha512"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/manifest"
	"github.com/docker/distribution/reference"
	"github.com/docker/libtrust"
	"github.com/opencontainers/go-digest"
)

type diffID digest.Digest

// gzippedEmptyTar is a gzip-compressed version of an empty tar file
// (1024 NULL bytes)
var gzippedEmptyTar = []byte{
	31, 139, 8, 0, 0, 9, 110, 136, 0, 255, 98, 24, 5, 163, 96, 20, 140, 88,
	0, 8, 0, 0, 255, 255, 46, 175, 181, 239, 0, 4, 0, 0,
}

// digestSHA256GzippedEmptyTar is the canonical sha256 digest of
// gzippedEmptyTar
const digestSHA256GzippedEmptyTar = digest.Digest("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")

// configManifestBuilder is a type for constructing manifests from an image
// configuration and generic descriptors.
type configManifestBuilder struct {
	// bs is a BlobService used to create empty layer tars in the
	// blob store if necessary.
	bs distribution.BlobService
	// pk is the libtrust private key used to sign the final manifest.
	pk libtrust.PrivateKey
	// configJSON is configuration supplied when the ManifestBuilder was
	// created.
	configJSON []byte
	// ref contains the name and optional tag provided to NewConfigManifestBuilder.
	ref reference.Named
	// descriptors is the set of descriptors referencing the layers.
	descriptors []distribution.Descriptor
	// emptyTarDigest is set to a valid digest if an empty tar has been
	// put in the blob store; otherwise it is empty.
	emptyTarDigest digest.Digest
}

// NewConfigManifestBuilder is used to build new manifests for the current
// schema version from an image configuration and a set of descriptors.
// It takes a BlobService so that it can add an empty tar to the blob store
// if the resulting manifest needs empty layers.
func NewConfigManifestBuilder(bs distribution.BlobService, pk libtrust.PrivateKey, ref reference.Named, configJSON []byte) distribution.ManifestBuilder {
	return &configManifestBuilder{
		bs:         bs,
		pk:         pk,
		configJSON: configJSON,
		ref:        ref,
	}
}

// Build produces a final manifest from the given references
func (mb *configManifestBuilder) Build(ctx context.Context) (m distribution.Manifest, err error) {
	type imageRootFS struct {
		Type      string   `json:"type"`
		DiffIDs   []diffID `json:"diff_ids,omitempty"`
		BaseLayer string   `json:"base_layer,omitempty"`
	}

	type imageHistory struct {
		Created    time.Time `json:"created"`
		Author     string    `json:"author,omitempty"`
		CreatedBy  string    `json:"created_by,omitempty"`
		Comment    string    `json:"comment,omitempty"`
		EmptyLayer bool      `json:"empty_layer,omitempty"`
	}

	type imageConfig struct {
		RootFS       *imageRootFS   `json:"rootfs,omitempty"`
		History      []imageHistory `json:"history,omitempty"`
		Architecture string         `json:"architecture,omitempty"`
	}

	var img imageConfig

	if err := json.Unmarshal(mb.configJSON, &img); err != nil {
		return nil, err
	}

	if len(img.History) == 0 {
		return nil, errors.New("empty history when trying to create schema1 manifest")
	}

	if len(img.RootFS.DiffIDs) != len(mb.descriptors) {
		return nil, fmt.Errorf("number of descriptors and number of layers in rootfs must match: len(%v) != len(%v)", img.RootFS.DiffIDs, mb.descriptors)
	}

	// Generate IDs for each layer
	// For non-top-level layers, create fake V1Compatibility strings that
	// fit the format and don't collide with anything else, but don't
	// result in runnable images on their own.
	type v1Compatibility struct {
		ID              string    `json:"id"`
		Parent          string    `json:"parent,omitempty"`
		Comment         string    `json:"comment,omitempty"`
		Created         time.Time `json:"created"`
		ContainerConfig struct {
			Cmd []string
		} `json:"container_config,omitempty"`
		Author    string `json:"author,omitempty"`
		ThrowAway bool   `json:"throwaway,omitempty"`
	}

	fsLayerList := make([]FSLayer, len(img.History))
	history := make([]History, len(img.History))

	parent := ""
	layerCounter := 0
	for i, h := range img.History[:len(img.History)-1] {
		var blobsum digest.Digest
		if h.EmptyLayer {
			if blobsum, err = mb.emptyTar(ctx); err != nil {
				return nil, err
			}
		} else {
			if len(img.RootFS.DiffIDs) <= layerCounter {
				return nil, errors.New("too many non-empty layers in History section")
			}
			blobsum = mb.descriptors[layerCounter].Digest
			layerCounter++
		}

		v1ID := digest.FromBytes([]byte(blobsum.Hex() + " " + parent)).Hex()

		if i == 0 && img.RootFS.BaseLayer != "" {
			// windows-only baselayer setup
			baseID := sha512.Sum384([]byte(img.RootFS.BaseLayer))
			parent = fmt.Sprintf("%x", baseID[:32])
		}

		v1Compatibility := v1Compatibility{
			ID:      v1ID,
			Parent:  parent,
			Comment: h.Comment,
			Created: h.Created,
			Author:  h.Author,
		}
		v1Compatibility.ContainerConfig.Cmd = []string{img.History[i].CreatedBy}
		if h.EmptyLayer {
			v1Compatibility.ThrowAway = true
		}
		jsonBytes, err := json.Marshal(&v1Compatibility)
		if err != nil {
			return nil, err
		}

		reversedIndex := len(img.History) - i - 1
		history[reversedIndex].V1Compatibility = string(jsonBytes)
		fsLayerList[reversedIndex] = FSLayer{BlobSum: blobsum}

		parent = v1ID
	}

	latestHistory := img.History[len(img.History)-1]

	var blobsum digest.Digest
	if latestHistory.EmptyLayer {
		if blobsum, err = mb.emptyTar(ctx); err != nil {
			return nil, err
		}
	} else {
		if len(img.RootFS.DiffIDs) <= layerCounter {
			return nil, errors.New("too many non-empty layers in History section")
		}
		blobsum = mb.descriptors[layerCounter].Digest
	}

	fsLayerList[0] = FSLayer{BlobSum: blobsum}
	dgst := digest.FromBytes([]byte(blobsum.Hex() + " " + parent + " " + string(mb.configJSON)))

	// Top-level v1compatibility string should be a modified version of the
	// image config.
	transformedConfig, err := MakeV1ConfigFromConfig(mb.configJSON, dgst.Hex(), parent, latestHistory.EmptyLayer)
	if err != nil {
		return nil, err
	}

	history[0].V1Compatibility = string(transformedConfig)

	tag := ""
	if tagged, isTagged := mb.ref.(reference.Tagged); isTagged {
		tag = tagged.Tag()
	}

	mfst := Manifest{
		Versioned: manifest.Versioned{
			SchemaVersion: 1,
		},
		Name:         mb.ref.Name(),
		Tag:          tag,
		Architecture: img.Architecture,
		FSLayers:     fsLayerList,
		History:      history,
	}

	return Sign(&mfst, mb.pk)
}

// emptyTar pushes a compressed empty tar to the blob store if one doesn't
// already exist, and returns its blobsum.
func (mb *configManifestBuilder) emptyTar(ctx context.Context) (digest.Digest, error) {
	if mb.emptyTarDigest != "" {
		// Already put an empty tar
		return mb.emptyTarDigest, nil
	}

	descriptor, err := mb.bs.Stat(ctx, digestSHA256GzippedEmptyTar)
	switch err {
	case nil:
		mb.emptyTarDigest = descriptor.Digest
		return descriptor.Digest, nil
	case distribution.ErrBlobUnknown:
		// nop
	default:
		return "", err
	}

	// Add gzipped empty tar to the blob store
	descriptor, err = mb.bs.Put(ctx, "", gzippedEmptyTar)
	if err != nil {
		return "", err
	}

	mb.emptyTarDigest = descriptor.Digest

	return descriptor.Digest, nil
}

// AppendReference adds a reference to the current ManifestBuilder
func (mb *configManifestBuilder) AppendReference(d distribution.Describable) error {
	descriptor := d.Descriptor()

	if err := descriptor.Digest.Validate(); err != nil {
		return err
	}

	mb.descriptors = append(mb.descriptors, descriptor)
	return nil
}

// References returns the current references added to this builder
func (mb *configManifestBuilder) References() []distribution.Descriptor {
	return mb.descriptors
}

// MakeV1ConfigFromConfig creates an legacy V1 image config from image config JSON
func MakeV1ConfigFromConfig(configJSON []byte, v1ID, parentV1ID string, throwaway bool) ([]byte, error) {
	// Top-level v1compatibility string should be a modified version of the
	// image config.
	var configAsMap map[string]*json.RawMessage
	if err := json.Unmarshal(configJSON, &configAsMap); err != nil {
		return nil, err
	}

	// Delete fields that didn't exist in old manifest
	delete(configAsMap, "rootfs")
	delete(configAsMap, "history")
	configAsMap["id"] = rawJSON(v1ID)
	if parentV1ID != "" {
		configAsMap["parent"] = rawJSON(parentV1ID)
	}
	if throwaway {
		configAsMap["throwaway"] = rawJSON(true)
	}

	return json.Marshal(configAsMap)
}

func rawJSON(value interface{}) *json.RawMessage {
	jsonval, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return (*json.RawMessage)(&jsonval)
}
