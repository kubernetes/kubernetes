package storage

import (
	"fmt"

	"encoding/json"
	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/manifest/schema2"
)

//schema2ManifestHandler is a ManifestHandler that covers schema2 manifests.
type schema2ManifestHandler struct {
	repository *repository
	blobStore  *linkedBlobStore
	ctx        context.Context
}

var _ ManifestHandler = &schema2ManifestHandler{}

func (ms *schema2ManifestHandler) Unmarshal(ctx context.Context, dgst digest.Digest, content []byte) (distribution.Manifest, error) {
	context.GetLogger(ms.ctx).Debug("(*schema2ManifestHandler).Unmarshal")

	var m schema2.DeserializedManifest
	if err := json.Unmarshal(content, &m); err != nil {
		return nil, err
	}

	return &m, nil
}

func (ms *schema2ManifestHandler) Put(ctx context.Context, manifest distribution.Manifest, skipDependencyVerification bool) (digest.Digest, error) {
	context.GetLogger(ms.ctx).Debug("(*schema2ManifestHandler).Put")

	m, ok := manifest.(*schema2.DeserializedManifest)
	if !ok {
		return "", fmt.Errorf("non-schema2 manifest put to schema2ManifestHandler: %T", manifest)
	}

	if err := ms.verifyManifest(ms.ctx, *m, skipDependencyVerification); err != nil {
		return "", err
	}

	mt, payload, err := m.Payload()
	if err != nil {
		return "", err
	}

	revision, err := ms.blobStore.Put(ctx, mt, payload)
	if err != nil {
		context.GetLogger(ctx).Errorf("error putting payload into blobstore: %v", err)
		return "", err
	}

	// Link the revision into the repository.
	if err := ms.blobStore.linkBlob(ctx, revision); err != nil {
		return "", err
	}

	return revision.Digest, nil
}

// verifyManifest ensures that the manifest content is valid from the
// perspective of the registry. As a policy, the registry only tries to store
// valid content, leaving trust policies of that content up to consumers.
func (ms *schema2ManifestHandler) verifyManifest(ctx context.Context, mnfst schema2.DeserializedManifest, skipDependencyVerification bool) error {
	var errs distribution.ErrManifestVerification

	if !skipDependencyVerification {
		target := mnfst.Target()
		_, err := ms.repository.Blobs(ctx).Stat(ctx, target.Digest)
		if err != nil {
			if err != distribution.ErrBlobUnknown {
				errs = append(errs, err)
			}

			// On error here, we always append unknown blob errors.
			errs = append(errs, distribution.ErrManifestBlobUnknown{Digest: target.Digest})
		}

		for _, fsLayer := range mnfst.References() {
			_, err := ms.repository.Blobs(ctx).Stat(ctx, fsLayer.Digest)
			if err != nil {
				if err != distribution.ErrBlobUnknown {
					errs = append(errs, err)
				}

				// On error here, we always append unknown blob errors.
				errs = append(errs, distribution.ErrManifestBlobUnknown{Digest: fsLayer.Digest})
			}
		}
	}
	if len(errs) != 0 {
		return errs
	}

	return nil
}
