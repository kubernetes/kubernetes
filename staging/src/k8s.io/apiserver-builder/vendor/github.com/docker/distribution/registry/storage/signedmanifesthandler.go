package storage

import (
	"encoding/json"
	"fmt"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/reference"
	"github.com/docker/libtrust"
)

// signedManifestHandler is a ManifestHandler that covers schema1 manifests. It
// can unmarshal and put schema1 manifests that have been signed by libtrust.
type signedManifestHandler struct {
	repository *repository
	blobStore  *linkedBlobStore
	ctx        context.Context
	signatures *signatureStore
}

var _ ManifestHandler = &signedManifestHandler{}

func (ms *signedManifestHandler) Unmarshal(ctx context.Context, dgst digest.Digest, content []byte) (distribution.Manifest, error) {
	context.GetLogger(ms.ctx).Debug("(*signedManifestHandler).Unmarshal")

	var (
		signatures [][]byte
		err        error
	)
	if ms.repository.schema1SignaturesEnabled {
		// Fetch the signatures for the manifest
		signatures, err = ms.signatures.Get(dgst)
		if err != nil {
			return nil, err
		}
	}

	jsig, err := libtrust.NewJSONSignature(content, signatures...)
	if err != nil {
		return nil, err
	}

	if ms.repository.schema1SigningKey != nil {
		if err := jsig.Sign(ms.repository.schema1SigningKey); err != nil {
			return nil, err
		}
	} else if !ms.repository.schema1SignaturesEnabled {
		return nil, fmt.Errorf("missing signing key with signature store disabled")
	}

	// Extract the pretty JWS
	raw, err := jsig.PrettySignature("signatures")
	if err != nil {
		return nil, err
	}

	var sm schema1.SignedManifest
	if err := json.Unmarshal(raw, &sm); err != nil {
		return nil, err
	}
	return &sm, nil
}

func (ms *signedManifestHandler) Put(ctx context.Context, manifest distribution.Manifest, skipDependencyVerification bool) (digest.Digest, error) {
	context.GetLogger(ms.ctx).Debug("(*signedManifestHandler).Put")

	sm, ok := manifest.(*schema1.SignedManifest)
	if !ok {
		return "", fmt.Errorf("non-schema1 manifest put to signedManifestHandler: %T", manifest)
	}

	if err := ms.verifyManifest(ms.ctx, *sm, skipDependencyVerification); err != nil {
		return "", err
	}

	mt := schema1.MediaTypeManifest
	payload := sm.Canonical

	revision, err := ms.blobStore.Put(ctx, mt, payload)
	if err != nil {
		context.GetLogger(ctx).Errorf("error putting payload into blobstore: %v", err)
		return "", err
	}

	// Link the revision into the repository.
	if err := ms.blobStore.linkBlob(ctx, revision); err != nil {
		return "", err
	}

	if ms.repository.schema1SignaturesEnabled {
		// Grab each json signature and store them.
		signatures, err := sm.Signatures()
		if err != nil {
			return "", err
		}

		if err := ms.signatures.Put(revision.Digest, signatures...); err != nil {
			return "", err
		}
	}

	return revision.Digest, nil
}

// verifyManifest ensures that the manifest content is valid from the
// perspective of the registry. It ensures that the signature is valid for the
// enclosed payload. As a policy, the registry only tries to store valid
// content, leaving trust policies of that content up to consumers.
func (ms *signedManifestHandler) verifyManifest(ctx context.Context, mnfst schema1.SignedManifest, skipDependencyVerification bool) error {
	var errs distribution.ErrManifestVerification

	if len(mnfst.Name) > reference.NameTotalLengthMax {
		errs = append(errs,
			distribution.ErrManifestNameInvalid{
				Name:   mnfst.Name,
				Reason: fmt.Errorf("manifest name must not be more than %v characters", reference.NameTotalLengthMax),
			})
	}

	if !reference.NameRegexp.MatchString(mnfst.Name) {
		errs = append(errs,
			distribution.ErrManifestNameInvalid{
				Name:   mnfst.Name,
				Reason: fmt.Errorf("invalid manifest name format"),
			})
	}

	if len(mnfst.History) != len(mnfst.FSLayers) {
		errs = append(errs, fmt.Errorf("mismatched history and fslayer cardinality %d != %d",
			len(mnfst.History), len(mnfst.FSLayers)))
	}

	if _, err := schema1.Verify(&mnfst); err != nil {
		switch err {
		case libtrust.ErrMissingSignatureKey, libtrust.ErrInvalidJSONContent, libtrust.ErrMissingSignatureKey:
			errs = append(errs, distribution.ErrManifestUnverified{})
		default:
			if err.Error() == "invalid signature" { // TODO(stevvooe): This should be exported by libtrust
				errs = append(errs, distribution.ErrManifestUnverified{})
			} else {
				errs = append(errs, err)
			}
		}
	}

	if !skipDependencyVerification {
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
