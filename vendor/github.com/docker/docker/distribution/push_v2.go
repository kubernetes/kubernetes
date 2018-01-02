package distribution

import (
	"errors"
	"fmt"
	"io"
	"runtime"
	"sort"
	"strings"
	"sync"

	"golang.org/x/net/context"

	"github.com/docker/distribution"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/client"
	apitypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/distribution/xfer"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/registry"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
)

const (
	smallLayerMaximumSize  = 100 * (1 << 10) // 100KB
	middleLayerMaximumSize = 10 * (1 << 20)  // 10MB
)

type v2Pusher struct {
	v2MetadataService metadata.V2MetadataService
	ref               reference.Named
	endpoint          registry.APIEndpoint
	repoInfo          *registry.RepositoryInfo
	config            *ImagePushConfig
	repo              distribution.Repository

	// pushState is state built by the Upload functions.
	pushState pushState
}

type pushState struct {
	sync.Mutex
	// remoteLayers is the set of layers known to exist on the remote side.
	// This avoids redundant queries when pushing multiple tags that
	// involve the same layers. It is also used to fill in digest and size
	// information when building the manifest.
	remoteLayers map[layer.DiffID]distribution.Descriptor
	// confirmedV2 is set to true if we confirm we're talking to a v2
	// registry. This is used to limit fallbacks to the v1 protocol.
	confirmedV2 bool
}

func (p *v2Pusher) Push(ctx context.Context) (err error) {
	p.pushState.remoteLayers = make(map[layer.DiffID]distribution.Descriptor)

	p.repo, p.pushState.confirmedV2, err = NewV2Repository(ctx, p.repoInfo, p.endpoint, p.config.MetaHeaders, p.config.AuthConfig, "push", "pull")
	if err != nil {
		logrus.Debugf("Error getting v2 registry: %v", err)
		return err
	}

	if err = p.pushV2Repository(ctx); err != nil {
		if continueOnError(err) {
			return fallbackError{
				err:         err,
				confirmedV2: p.pushState.confirmedV2,
				transportOK: true,
			}
		}
	}
	return err
}

func (p *v2Pusher) pushV2Repository(ctx context.Context) (err error) {
	if namedTagged, isNamedTagged := p.ref.(reference.NamedTagged); isNamedTagged {
		imageID, err := p.config.ReferenceStore.Get(p.ref)
		if err != nil {
			return fmt.Errorf("tag does not exist: %s", reference.FamiliarString(p.ref))
		}

		return p.pushV2Tag(ctx, namedTagged, imageID)
	}

	if !reference.IsNameOnly(p.ref) {
		return errors.New("cannot push a digest reference")
	}

	// Pull all tags
	pushed := 0
	for _, association := range p.config.ReferenceStore.ReferencesByName(p.ref) {
		if namedTagged, isNamedTagged := association.Ref.(reference.NamedTagged); isNamedTagged {
			pushed++
			if err := p.pushV2Tag(ctx, namedTagged, association.ID); err != nil {
				return err
			}
		}
	}

	if pushed == 0 {
		return fmt.Errorf("no tags to push for %s", reference.FamiliarName(p.repoInfo.Name))
	}

	return nil
}

func (p *v2Pusher) pushV2Tag(ctx context.Context, ref reference.NamedTagged, id digest.Digest) error {
	logrus.Debugf("Pushing repository: %s", reference.FamiliarString(ref))

	imgConfig, err := p.config.ImageStore.Get(id)
	if err != nil {
		return fmt.Errorf("could not find image from tag %s: %v", reference.FamiliarString(ref), err)
	}

	rootfs, _, err := p.config.ImageStore.RootFSAndPlatformFromConfig(imgConfig)
	if err != nil {
		return fmt.Errorf("unable to get rootfs for image %s: %s", reference.FamiliarString(ref), err)
	}

	l, err := p.config.LayerStore.Get(rootfs.ChainID())
	if err != nil {
		return fmt.Errorf("failed to get top layer from image: %v", err)
	}
	defer l.Release()

	hmacKey, err := metadata.ComputeV2MetadataHMACKey(p.config.AuthConfig)
	if err != nil {
		return fmt.Errorf("failed to compute hmac key of auth config: %v", err)
	}

	var descriptors []xfer.UploadDescriptor

	descriptorTemplate := v2PushDescriptor{
		v2MetadataService: p.v2MetadataService,
		hmacKey:           hmacKey,
		repoInfo:          p.repoInfo.Name,
		ref:               p.ref,
		endpoint:          p.endpoint,
		repo:              p.repo,
		pushState:         &p.pushState,
	}

	// Loop bounds condition is to avoid pushing the base layer on Windows.
	for range rootfs.DiffIDs {
		descriptor := descriptorTemplate
		descriptor.layer = l
		descriptor.checkedDigests = make(map[digest.Digest]struct{})
		descriptors = append(descriptors, &descriptor)

		l = l.Parent()
	}

	if err := p.config.UploadManager.Upload(ctx, descriptors, p.config.ProgressOutput); err != nil {
		return err
	}

	// Try schema2 first
	builder := schema2.NewManifestBuilder(p.repo.Blobs(ctx), p.config.ConfigMediaType, imgConfig)
	manifest, err := manifestFromBuilder(ctx, builder, descriptors)
	if err != nil {
		return err
	}

	manSvc, err := p.repo.Manifests(ctx)
	if err != nil {
		return err
	}

	putOptions := []distribution.ManifestServiceOption{distribution.WithTag(ref.Tag())}
	if _, err = manSvc.Put(ctx, manifest, putOptions...); err != nil {
		if runtime.GOOS == "windows" || p.config.TrustKey == nil || p.config.RequireSchema2 {
			logrus.Warnf("failed to upload schema2 manifest: %v", err)
			return err
		}

		logrus.Warnf("failed to upload schema2 manifest: %v - falling back to schema1", err)

		manifestRef, err := reference.WithTag(p.repo.Named(), ref.Tag())
		if err != nil {
			return err
		}
		builder = schema1.NewConfigManifestBuilder(p.repo.Blobs(ctx), p.config.TrustKey, manifestRef, imgConfig)
		manifest, err = manifestFromBuilder(ctx, builder, descriptors)
		if err != nil {
			return err
		}

		if _, err = manSvc.Put(ctx, manifest, putOptions...); err != nil {
			return err
		}
	}

	var canonicalManifest []byte

	switch v := manifest.(type) {
	case *schema1.SignedManifest:
		canonicalManifest = v.Canonical
	case *schema2.DeserializedManifest:
		_, canonicalManifest, err = v.Payload()
		if err != nil {
			return err
		}
	}

	manifestDigest := digest.FromBytes(canonicalManifest)
	progress.Messagef(p.config.ProgressOutput, "", "%s: digest: %s size: %d", ref.Tag(), manifestDigest, len(canonicalManifest))

	if err := addDigestReference(p.config.ReferenceStore, ref, manifestDigest, id); err != nil {
		return err
	}

	// Signal digest to the trust client so it can sign the
	// push, if appropriate.
	progress.Aux(p.config.ProgressOutput, apitypes.PushResult{Tag: ref.Tag(), Digest: manifestDigest.String(), Size: len(canonicalManifest)})

	return nil
}

func manifestFromBuilder(ctx context.Context, builder distribution.ManifestBuilder, descriptors []xfer.UploadDescriptor) (distribution.Manifest, error) {
	// descriptors is in reverse order; iterate backwards to get references
	// appended in the right order.
	for i := len(descriptors) - 1; i >= 0; i-- {
		if err := builder.AppendReference(descriptors[i].(*v2PushDescriptor)); err != nil {
			return nil, err
		}
	}

	return builder.Build(ctx)
}

type v2PushDescriptor struct {
	layer             PushLayer
	v2MetadataService metadata.V2MetadataService
	hmacKey           []byte
	repoInfo          reference.Named
	ref               reference.Named
	endpoint          registry.APIEndpoint
	repo              distribution.Repository
	pushState         *pushState
	remoteDescriptor  distribution.Descriptor
	// a set of digests whose presence has been checked in a target repository
	checkedDigests map[digest.Digest]struct{}
}

func (pd *v2PushDescriptor) Key() string {
	return "v2push:" + pd.ref.Name() + " " + pd.layer.DiffID().String()
}

func (pd *v2PushDescriptor) ID() string {
	return stringid.TruncateID(pd.layer.DiffID().String())
}

func (pd *v2PushDescriptor) DiffID() layer.DiffID {
	return pd.layer.DiffID()
}

func (pd *v2PushDescriptor) Upload(ctx context.Context, progressOutput progress.Output) (distribution.Descriptor, error) {
	// Skip foreign layers unless this registry allows nondistributable artifacts.
	if !pd.endpoint.AllowNondistributableArtifacts {
		if fs, ok := pd.layer.(distribution.Describable); ok {
			if d := fs.Descriptor(); len(d.URLs) > 0 {
				progress.Update(progressOutput, pd.ID(), "Skipped foreign layer")
				return d, nil
			}
		}
	}

	diffID := pd.DiffID()

	pd.pushState.Lock()
	if descriptor, ok := pd.pushState.remoteLayers[diffID]; ok {
		// it is already known that the push is not needed and
		// therefore doing a stat is unnecessary
		pd.pushState.Unlock()
		progress.Update(progressOutput, pd.ID(), "Layer already exists")
		return descriptor, nil
	}
	pd.pushState.Unlock()

	maxMountAttempts, maxExistenceChecks, checkOtherRepositories := getMaxMountAndExistenceCheckAttempts(pd.layer)

	// Do we have any metadata associated with this layer's DiffID?
	v2Metadata, err := pd.v2MetadataService.GetMetadata(diffID)
	if err == nil {
		// check for blob existence in the target repository
		descriptor, exists, err := pd.layerAlreadyExists(ctx, progressOutput, diffID, true, 1, v2Metadata)
		if exists || err != nil {
			return descriptor, err
		}
	}

	// if digest was empty or not saved, or if blob does not exist on the remote repository,
	// then push the blob.
	bs := pd.repo.Blobs(ctx)

	var layerUpload distribution.BlobWriter

	// Attempt to find another repository in the same registry to mount the layer from to avoid an unnecessary upload
	candidates := getRepositoryMountCandidates(pd.repoInfo, pd.hmacKey, maxMountAttempts, v2Metadata)
	for _, mountCandidate := range candidates {
		logrus.Debugf("attempting to mount layer %s (%s) from %s", diffID, mountCandidate.Digest, mountCandidate.SourceRepository)
		createOpts := []distribution.BlobCreateOption{}

		if len(mountCandidate.SourceRepository) > 0 {
			namedRef, err := reference.ParseNormalizedNamed(mountCandidate.SourceRepository)
			if err != nil {
				logrus.Errorf("failed to parse source repository reference %v: %v", reference.FamiliarString(namedRef), err)
				pd.v2MetadataService.Remove(mountCandidate)
				continue
			}

			// Candidates are always under same domain, create remote reference
			// with only path to set mount from with
			remoteRef, err := reference.WithName(reference.Path(namedRef))
			if err != nil {
				logrus.Errorf("failed to make remote reference out of %q: %v", reference.Path(namedRef), err)
				continue
			}

			canonicalRef, err := reference.WithDigest(reference.TrimNamed(remoteRef), mountCandidate.Digest)
			if err != nil {
				logrus.Errorf("failed to make canonical reference: %v", err)
				continue
			}

			createOpts = append(createOpts, client.WithMountFrom(canonicalRef))
		}

		// send the layer
		lu, err := bs.Create(ctx, createOpts...)
		switch err := err.(type) {
		case nil:
			// noop
		case distribution.ErrBlobMounted:
			progress.Updatef(progressOutput, pd.ID(), "Mounted from %s", err.From.Name())

			err.Descriptor.MediaType = schema2.MediaTypeLayer

			pd.pushState.Lock()
			pd.pushState.confirmedV2 = true
			pd.pushState.remoteLayers[diffID] = err.Descriptor
			pd.pushState.Unlock()

			// Cache mapping from this layer's DiffID to the blobsum
			if err := pd.v2MetadataService.TagAndAdd(diffID, pd.hmacKey, metadata.V2Metadata{
				Digest:           err.Descriptor.Digest,
				SourceRepository: pd.repoInfo.Name(),
			}); err != nil {
				return distribution.Descriptor{}, xfer.DoNotRetry{Err: err}
			}
			return err.Descriptor, nil
		default:
			logrus.Infof("failed to mount layer %s (%s) from %s: %v", diffID, mountCandidate.Digest, mountCandidate.SourceRepository, err)
		}

		if len(mountCandidate.SourceRepository) > 0 &&
			(metadata.CheckV2MetadataHMAC(&mountCandidate, pd.hmacKey) ||
				len(mountCandidate.HMAC) == 0) {
			cause := "blob mount failure"
			if err != nil {
				cause = fmt.Sprintf("an error: %v", err.Error())
			}
			logrus.Debugf("removing association between layer %s and %s due to %s", mountCandidate.Digest, mountCandidate.SourceRepository, cause)
			pd.v2MetadataService.Remove(mountCandidate)
		}

		if lu != nil {
			// cancel previous upload
			cancelLayerUpload(ctx, mountCandidate.Digest, layerUpload)
			layerUpload = lu
		}
	}

	if maxExistenceChecks-len(pd.checkedDigests) > 0 {
		// do additional layer existence checks with other known digests if any
		descriptor, exists, err := pd.layerAlreadyExists(ctx, progressOutput, diffID, checkOtherRepositories, maxExistenceChecks-len(pd.checkedDigests), v2Metadata)
		if exists || err != nil {
			return descriptor, err
		}
	}

	logrus.Debugf("Pushing layer: %s", diffID)
	if layerUpload == nil {
		layerUpload, err = bs.Create(ctx)
		if err != nil {
			return distribution.Descriptor{}, retryOnError(err)
		}
	}
	defer layerUpload.Close()

	// upload the blob
	desc, err := pd.uploadUsingSession(ctx, progressOutput, diffID, layerUpload)
	if err != nil {
		return desc, err
	}

	return desc, nil
}

func (pd *v2PushDescriptor) SetRemoteDescriptor(descriptor distribution.Descriptor) {
	pd.remoteDescriptor = descriptor
}

func (pd *v2PushDescriptor) Descriptor() distribution.Descriptor {
	return pd.remoteDescriptor
}

func (pd *v2PushDescriptor) uploadUsingSession(
	ctx context.Context,
	progressOutput progress.Output,
	diffID layer.DiffID,
	layerUpload distribution.BlobWriter,
) (distribution.Descriptor, error) {
	var reader io.ReadCloser

	contentReader, err := pd.layer.Open()
	if err != nil {
		return distribution.Descriptor{}, retryOnError(err)
	}

	size, _ := pd.layer.Size()

	reader = progress.NewProgressReader(ioutils.NewCancelReadCloser(ctx, contentReader), progressOutput, size, pd.ID(), "Pushing")

	switch m := pd.layer.MediaType(); m {
	case schema2.MediaTypeUncompressedLayer:
		compressedReader, compressionDone := compress(reader)
		defer func(closer io.Closer) {
			closer.Close()
			<-compressionDone
		}(reader)
		reader = compressedReader
	case schema2.MediaTypeLayer:
	default:
		reader.Close()
		return distribution.Descriptor{}, fmt.Errorf("unsupported layer media type %s", m)
	}

	digester := digest.Canonical.Digester()
	tee := io.TeeReader(reader, digester.Hash())

	nn, err := layerUpload.ReadFrom(tee)
	reader.Close()
	if err != nil {
		return distribution.Descriptor{}, retryOnError(err)
	}

	pushDigest := digester.Digest()
	if _, err := layerUpload.Commit(ctx, distribution.Descriptor{Digest: pushDigest}); err != nil {
		return distribution.Descriptor{}, retryOnError(err)
	}

	logrus.Debugf("uploaded layer %s (%s), %d bytes", diffID, pushDigest, nn)
	progress.Update(progressOutput, pd.ID(), "Pushed")

	// Cache mapping from this layer's DiffID to the blobsum
	if err := pd.v2MetadataService.TagAndAdd(diffID, pd.hmacKey, metadata.V2Metadata{
		Digest:           pushDigest,
		SourceRepository: pd.repoInfo.Name(),
	}); err != nil {
		return distribution.Descriptor{}, xfer.DoNotRetry{Err: err}
	}

	desc := distribution.Descriptor{
		Digest:    pushDigest,
		MediaType: schema2.MediaTypeLayer,
		Size:      nn,
	}

	pd.pushState.Lock()
	// If Commit succeeded, that's an indication that the remote registry speaks the v2 protocol.
	pd.pushState.confirmedV2 = true
	pd.pushState.remoteLayers[diffID] = desc
	pd.pushState.Unlock()

	return desc, nil
}

// layerAlreadyExists checks if the registry already knows about any of the metadata passed in the "metadata"
// slice. If it finds one that the registry knows about, it returns the known digest and "true". If
// "checkOtherRepositories" is true, stat will be performed also with digests mapped to any other repository
// (not just the target one).
func (pd *v2PushDescriptor) layerAlreadyExists(
	ctx context.Context,
	progressOutput progress.Output,
	diffID layer.DiffID,
	checkOtherRepositories bool,
	maxExistenceCheckAttempts int,
	v2Metadata []metadata.V2Metadata,
) (desc distribution.Descriptor, exists bool, err error) {
	// filter the metadata
	candidates := []metadata.V2Metadata{}
	for _, meta := range v2Metadata {
		if len(meta.SourceRepository) > 0 && !checkOtherRepositories && meta.SourceRepository != pd.repoInfo.Name() {
			continue
		}
		candidates = append(candidates, meta)
	}
	// sort the candidates by similarity
	sortV2MetadataByLikenessAndAge(pd.repoInfo, pd.hmacKey, candidates)

	digestToMetadata := make(map[digest.Digest]*metadata.V2Metadata)
	// an array of unique blob digests ordered from the best mount candidates to worst
	layerDigests := []digest.Digest{}
	for i := 0; i < len(candidates); i++ {
		if len(layerDigests) >= maxExistenceCheckAttempts {
			break
		}
		meta := &candidates[i]
		if _, exists := digestToMetadata[meta.Digest]; exists {
			// keep reference just to the first mapping (the best mount candidate)
			continue
		}
		if _, exists := pd.checkedDigests[meta.Digest]; exists {
			// existence of this digest has already been tested
			continue
		}
		digestToMetadata[meta.Digest] = meta
		layerDigests = append(layerDigests, meta.Digest)
	}

attempts:
	for _, dgst := range layerDigests {
		meta := digestToMetadata[dgst]
		logrus.Debugf("Checking for presence of layer %s (%s) in %s", diffID, dgst, pd.repoInfo.Name())
		desc, err = pd.repo.Blobs(ctx).Stat(ctx, dgst)
		pd.checkedDigests[meta.Digest] = struct{}{}
		switch err {
		case nil:
			if m, ok := digestToMetadata[desc.Digest]; !ok || m.SourceRepository != pd.repoInfo.Name() || !metadata.CheckV2MetadataHMAC(m, pd.hmacKey) {
				// cache mapping from this layer's DiffID to the blobsum
				if err := pd.v2MetadataService.TagAndAdd(diffID, pd.hmacKey, metadata.V2Metadata{
					Digest:           desc.Digest,
					SourceRepository: pd.repoInfo.Name(),
				}); err != nil {
					return distribution.Descriptor{}, false, xfer.DoNotRetry{Err: err}
				}
			}
			desc.MediaType = schema2.MediaTypeLayer
			exists = true
			break attempts
		case distribution.ErrBlobUnknown:
			if meta.SourceRepository == pd.repoInfo.Name() {
				// remove the mapping to the target repository
				pd.v2MetadataService.Remove(*meta)
			}
		default:
			logrus.WithError(err).Debugf("Failed to check for presence of layer %s (%s) in %s", diffID, dgst, pd.repoInfo.Name())
		}
	}

	if exists {
		progress.Update(progressOutput, pd.ID(), "Layer already exists")
		pd.pushState.Lock()
		pd.pushState.remoteLayers[diffID] = desc
		pd.pushState.Unlock()
	}

	return desc, exists, nil
}

// getMaxMountAndExistenceCheckAttempts returns a maximum number of cross repository mount attempts from
// source repositories of target registry, maximum number of layer existence checks performed on the target
// repository and whether the check shall be done also with digests mapped to different repositories. The
// decision is based on layer size. The smaller the layer, the fewer attempts shall be made because the cost
// of upload does not outweigh a latency.
func getMaxMountAndExistenceCheckAttempts(layer PushLayer) (maxMountAttempts, maxExistenceCheckAttempts int, checkOtherRepositories bool) {
	size, err := layer.Size()
	switch {
	// big blob
	case size > middleLayerMaximumSize:
		// 1st attempt to mount the blob few times
		// 2nd few existence checks with digests associated to any repository
		// then fallback to upload
		return 4, 3, true

	// middle sized blobs; if we could not get the size, assume we deal with middle sized blob
	case size > smallLayerMaximumSize, err != nil:
		// 1st attempt to mount blobs of average size few times
		// 2nd try at most 1 existence check if there's an existing mapping to the target repository
		// then fallback to upload
		return 3, 1, false

	// small blobs, do a minimum number of checks
	default:
		return 1, 1, false
	}
}

// getRepositoryMountCandidates returns an array of v2 metadata items belonging to the given registry. The
// array is sorted from youngest to oldest. If requireRegistryMatch is true, the resulting array will contain
// only metadata entries having registry part of SourceRepository matching the part of repoInfo.
func getRepositoryMountCandidates(
	repoInfo reference.Named,
	hmacKey []byte,
	max int,
	v2Metadata []metadata.V2Metadata,
) []metadata.V2Metadata {
	candidates := []metadata.V2Metadata{}
	for _, meta := range v2Metadata {
		sourceRepo, err := reference.ParseNamed(meta.SourceRepository)
		if err != nil || reference.Domain(repoInfo) != reference.Domain(sourceRepo) {
			continue
		}
		// target repository is not a viable candidate
		if meta.SourceRepository == repoInfo.Name() {
			continue
		}
		candidates = append(candidates, meta)
	}

	sortV2MetadataByLikenessAndAge(repoInfo, hmacKey, candidates)
	if max >= 0 && len(candidates) > max {
		// select the youngest metadata
		candidates = candidates[:max]
	}

	return candidates
}

// byLikeness is a sorting container for v2 metadata candidates for cross repository mount. The
// candidate "a" is preferred over "b":
//
//  1. if it was hashed using the same AuthConfig as the one used to authenticate to target repository and the
//     "b" was not
//  2. if a number of its repository path components exactly matching path components of target repository is higher
type byLikeness struct {
	arr            []metadata.V2Metadata
	hmacKey        []byte
	pathComponents []string
}

func (bla byLikeness) Less(i, j int) bool {
	aMacMatch := metadata.CheckV2MetadataHMAC(&bla.arr[i], bla.hmacKey)
	bMacMatch := metadata.CheckV2MetadataHMAC(&bla.arr[j], bla.hmacKey)
	if aMacMatch != bMacMatch {
		return aMacMatch
	}
	aMatch := numOfMatchingPathComponents(bla.arr[i].SourceRepository, bla.pathComponents)
	bMatch := numOfMatchingPathComponents(bla.arr[j].SourceRepository, bla.pathComponents)
	return aMatch > bMatch
}
func (bla byLikeness) Swap(i, j int) {
	bla.arr[i], bla.arr[j] = bla.arr[j], bla.arr[i]
}
func (bla byLikeness) Len() int { return len(bla.arr) }

func sortV2MetadataByLikenessAndAge(repoInfo reference.Named, hmacKey []byte, marr []metadata.V2Metadata) {
	// reverse the metadata array to shift the newest entries to the beginning
	for i := 0; i < len(marr)/2; i++ {
		marr[i], marr[len(marr)-i-1] = marr[len(marr)-i-1], marr[i]
	}
	// keep equal entries ordered from the youngest to the oldest
	sort.Stable(byLikeness{
		arr:            marr,
		hmacKey:        hmacKey,
		pathComponents: getPathComponents(repoInfo.Name()),
	})
}

// numOfMatchingPathComponents returns a number of path components in "pth" that exactly match "matchComponents".
func numOfMatchingPathComponents(pth string, matchComponents []string) int {
	pthComponents := getPathComponents(pth)
	i := 0
	for ; i < len(pthComponents) && i < len(matchComponents); i++ {
		if matchComponents[i] != pthComponents[i] {
			return i
		}
	}
	return i
}

func getPathComponents(path string) []string {
	return strings.Split(path, "/")
}

func cancelLayerUpload(ctx context.Context, dgst digest.Digest, layerUpload distribution.BlobWriter) {
	if layerUpload != nil {
		logrus.Debugf("cancelling upload of blob %s", dgst)
		err := layerUpload.Cancel(ctx)
		if err != nil {
			logrus.Warnf("failed to cancel upload: %v", err)
		}
	}
}
