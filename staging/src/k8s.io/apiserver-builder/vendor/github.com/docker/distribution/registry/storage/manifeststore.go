package storage

import (
	"fmt"
	"path"

	"encoding/json"
	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/manifest"
	"github.com/docker/distribution/manifest/manifestlist"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/distribution/registry/storage/driver"
)

// A ManifestHandler gets and puts manifests of a particular type.
type ManifestHandler interface {
	// Unmarshal unmarshals the manifest from a byte slice.
	Unmarshal(ctx context.Context, dgst digest.Digest, content []byte) (distribution.Manifest, error)

	// Put creates or updates the given manifest returning the manifest digest.
	Put(ctx context.Context, manifest distribution.Manifest, skipDependencyVerification bool) (digest.Digest, error)
}

// SkipLayerVerification allows a manifest to be Put before its
// layers are on the filesystem
func SkipLayerVerification() distribution.ManifestServiceOption {
	return skipLayerOption{}
}

type skipLayerOption struct{}

func (o skipLayerOption) Apply(m distribution.ManifestService) error {
	if ms, ok := m.(*manifestStore); ok {
		ms.skipDependencyVerification = true
		return nil
	}
	return fmt.Errorf("skip layer verification only valid for manifestStore")
}

type manifestStore struct {
	repository *repository
	blobStore  *linkedBlobStore
	ctx        context.Context

	skipDependencyVerification bool

	schema1Handler      ManifestHandler
	schema2Handler      ManifestHandler
	manifestListHandler ManifestHandler
}

var _ distribution.ManifestService = &manifestStore{}

func (ms *manifestStore) Exists(ctx context.Context, dgst digest.Digest) (bool, error) {
	context.GetLogger(ms.ctx).Debug("(*manifestStore).Exists")

	_, err := ms.blobStore.Stat(ms.ctx, dgst)
	if err != nil {
		if err == distribution.ErrBlobUnknown {
			return false, nil
		}

		return false, err
	}

	return true, nil
}

func (ms *manifestStore) Get(ctx context.Context, dgst digest.Digest, options ...distribution.ManifestServiceOption) (distribution.Manifest, error) {
	context.GetLogger(ms.ctx).Debug("(*manifestStore).Get")

	// TODO(stevvooe): Need to check descriptor from above to ensure that the
	// mediatype is as we expect for the manifest store.

	content, err := ms.blobStore.Get(ctx, dgst)
	if err != nil {
		if err == distribution.ErrBlobUnknown {
			return nil, distribution.ErrManifestUnknownRevision{
				Name:     ms.repository.Named().Name(),
				Revision: dgst,
			}
		}

		return nil, err
	}

	var versioned manifest.Versioned
	if err = json.Unmarshal(content, &versioned); err != nil {
		return nil, err
	}

	switch versioned.SchemaVersion {
	case 1:
		return ms.schema1Handler.Unmarshal(ctx, dgst, content)
	case 2:
		// This can be an image manifest or a manifest list
		switch versioned.MediaType {
		case schema2.MediaTypeManifest:
			return ms.schema2Handler.Unmarshal(ctx, dgst, content)
		case manifestlist.MediaTypeManifestList:
			return ms.manifestListHandler.Unmarshal(ctx, dgst, content)
		default:
			return nil, distribution.ErrManifestVerification{fmt.Errorf("unrecognized manifest content type %s", versioned.MediaType)}
		}
	}

	return nil, fmt.Errorf("unrecognized manifest schema version %d", versioned.SchemaVersion)
}

func (ms *manifestStore) Put(ctx context.Context, manifest distribution.Manifest, options ...distribution.ManifestServiceOption) (digest.Digest, error) {
	context.GetLogger(ms.ctx).Debug("(*manifestStore).Put")

	switch manifest.(type) {
	case *schema1.SignedManifest:
		return ms.schema1Handler.Put(ctx, manifest, ms.skipDependencyVerification)
	case *schema2.DeserializedManifest:
		return ms.schema2Handler.Put(ctx, manifest, ms.skipDependencyVerification)
	case *manifestlist.DeserializedManifestList:
		return ms.manifestListHandler.Put(ctx, manifest, ms.skipDependencyVerification)
	}

	return "", fmt.Errorf("unrecognized manifest type %T", manifest)
}

// Delete removes the revision of the specified manfiest.
func (ms *manifestStore) Delete(ctx context.Context, dgst digest.Digest) error {
	context.GetLogger(ms.ctx).Debug("(*manifestStore).Delete")
	return ms.blobStore.Delete(ctx, dgst)
}

func (ms *manifestStore) Enumerate(ctx context.Context, ingester func(digest.Digest) error) error {
	err := ms.blobStore.Enumerate(ctx, func(dgst digest.Digest) error {
		err := ingester(dgst)
		if err != nil {
			return err
		}
		return nil
	})
	return err
}

// Only valid for schema1 signed manifests
func (ms *manifestStore) GetSignatures(ctx context.Context, manifestDigest digest.Digest) ([]digest.Digest, error) {
	// sanity check that digest refers to a schema1 digest
	manifest, err := ms.Get(ctx, manifestDigest)
	if err != nil {
		return nil, err
	}

	if _, ok := manifest.(*schema1.SignedManifest); !ok {
		return nil, fmt.Errorf("digest %v is not for schema1 manifest", manifestDigest)
	}

	signaturesPath, err := pathFor(manifestSignaturesPathSpec{
		name:     ms.repository.Named().Name(),
		revision: manifestDigest,
	})
	if err != nil {
		return nil, err
	}

	var digests []digest.Digest
	alg := string(digest.SHA256)
	signaturePaths, err := ms.blobStore.driver.List(ctx, path.Join(signaturesPath, alg))

	switch err.(type) {
	case nil:
		break
	case driver.PathNotFoundError:
		// Manifest may have been pushed with signature store disabled
		return digests, nil
	default:
		return nil, err
	}

	for _, sigPath := range signaturePaths {
		sigdigest, err := digest.ParseDigest(alg + ":" + path.Base(sigPath))
		if err != nil {
			// merely found not a digest
			continue
		}
		digests = append(digests, sigdigest)
	}
	return digests, nil
}
