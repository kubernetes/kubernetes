package storage

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/opencontainers/go-digest"
)

var (
	errUnexpectedURL = errors.New("unexpected URL on layer")
	errMissingURL    = errors.New("missing URL on layer")
	errInvalidURL    = errors.New("invalid URL on layer")
)

//schema2ManifestHandler is a ManifestHandler that covers schema2 manifests.
type schema2ManifestHandler struct {
	repository   distribution.Repository
	blobStore    distribution.BlobStore
	ctx          context.Context
	manifestURLs manifestURLs
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

	return revision.Digest, nil
}

// verifyManifest ensures that the manifest content is valid from the
// perspective of the registry. As a policy, the registry only tries to store
// valid content, leaving trust policies of that content up to consumers.
func (ms *schema2ManifestHandler) verifyManifest(ctx context.Context, mnfst schema2.DeserializedManifest, skipDependencyVerification bool) error {
	var errs distribution.ErrManifestVerification

	if skipDependencyVerification {
		return nil
	}

	manifestService, err := ms.repository.Manifests(ctx)
	if err != nil {
		return err
	}

	blobsService := ms.repository.Blobs(ctx)

	for _, descriptor := range mnfst.References() {
		var err error

		switch descriptor.MediaType {
		case schema2.MediaTypeForeignLayer:
			// Clients download this layer from an external URL, so do not check for
			// its presense.
			if len(descriptor.URLs) == 0 {
				err = errMissingURL
			}
			allow := ms.manifestURLs.allow
			deny := ms.manifestURLs.deny
			for _, u := range descriptor.URLs {
				var pu *url.URL
				pu, err = url.Parse(u)
				if err != nil || (pu.Scheme != "http" && pu.Scheme != "https") || pu.Fragment != "" || (allow != nil && !allow.MatchString(u)) || (deny != nil && deny.MatchString(u)) {
					err = errInvalidURL
					break
				}
			}
		case schema2.MediaTypeManifest, schema1.MediaTypeManifest:
			var exists bool
			exists, err = manifestService.Exists(ctx, descriptor.Digest)
			if err != nil || !exists {
				err = distribution.ErrBlobUnknown // just coerce to unknown.
			}

			fallthrough // double check the blob store.
		default:
			// forward all else to blob storage
			if len(descriptor.URLs) == 0 {
				_, err = blobsService.Stat(ctx, descriptor.Digest)
			}
		}

		if err != nil {
			if err != distribution.ErrBlobUnknown {
				errs = append(errs, err)
			}

			// On error here, we always append unknown blob errors.
			errs = append(errs, distribution.ErrManifestBlobUnknown{Digest: descriptor.Digest})
		}
	}

	if len(errs) != 0 {
		return errs
	}

	return nil
}
