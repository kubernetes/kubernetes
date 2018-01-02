package distribution

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"runtime"

	"github.com/docker/distribution"
	"github.com/docker/distribution/manifest/manifestlist"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/client/auth"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/distribution/xfer"
	"github.com/docker/docker/image"
	"github.com/docker/docker/image/v1"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/system"
	refstore "github.com/docker/docker/reference"
	"github.com/docker/docker/registry"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

var (
	errRootFSMismatch = errors.New("layers from manifest don't match image configuration")
	errRootFSInvalid  = errors.New("invalid rootfs in image configuration")
)

// ImageConfigPullError is an error pulling the image config blob
// (only applies to schema2).
type ImageConfigPullError struct {
	Err error
}

// Error returns the error string for ImageConfigPullError.
func (e ImageConfigPullError) Error() string {
	return "error pulling image configuration: " + e.Err.Error()
}

type v2Puller struct {
	V2MetadataService metadata.V2MetadataService
	endpoint          registry.APIEndpoint
	config            *ImagePullConfig
	repoInfo          *registry.RepositoryInfo
	repo              distribution.Repository
	// confirmedV2 is set to true if we confirm we're talking to a v2
	// registry. This is used to limit fallbacks to the v1 protocol.
	confirmedV2 bool
}

func (p *v2Puller) Pull(ctx context.Context, ref reference.Named) (err error) {
	// TODO(tiborvass): was ReceiveTimeout
	p.repo, p.confirmedV2, err = NewV2Repository(ctx, p.repoInfo, p.endpoint, p.config.MetaHeaders, p.config.AuthConfig, "pull")
	if err != nil {
		logrus.Warnf("Error getting v2 registry: %v", err)
		return err
	}

	if err = p.pullV2Repository(ctx, ref); err != nil {
		if _, ok := err.(fallbackError); ok {
			return err
		}
		if continueOnError(err) {
			return fallbackError{
				err:         err,
				confirmedV2: p.confirmedV2,
				transportOK: true,
			}
		}
	}
	return err
}

func (p *v2Puller) pullV2Repository(ctx context.Context, ref reference.Named) (err error) {
	var layersDownloaded bool
	if !reference.IsNameOnly(ref) {
		layersDownloaded, err = p.pullV2Tag(ctx, ref)
		if err != nil {
			return err
		}
	} else {
		tags, err := p.repo.Tags(ctx).All(ctx)
		if err != nil {
			// If this repository doesn't exist on V2, we should
			// permit a fallback to V1.
			return allowV1Fallback(err)
		}

		// The v2 registry knows about this repository, so we will not
		// allow fallback to the v1 protocol even if we encounter an
		// error later on.
		p.confirmedV2 = true

		for _, tag := range tags {
			tagRef, err := reference.WithTag(ref, tag)
			if err != nil {
				return err
			}
			pulledNew, err := p.pullV2Tag(ctx, tagRef)
			if err != nil {
				// Since this is the pull-all-tags case, don't
				// allow an error pulling a particular tag to
				// make the whole pull fall back to v1.
				if fallbackErr, ok := err.(fallbackError); ok {
					return fallbackErr.err
				}
				return err
			}
			// pulledNew is true if either new layers were downloaded OR if existing images were newly tagged
			// TODO(tiborvass): should we change the name of `layersDownload`? What about message in WriteStatus?
			layersDownloaded = layersDownloaded || pulledNew
		}
	}

	writeStatus(reference.FamiliarString(ref), p.config.ProgressOutput, layersDownloaded)

	return nil
}

type v2LayerDescriptor struct {
	digest            digest.Digest
	diffID            layer.DiffID
	repoInfo          *registry.RepositoryInfo
	repo              distribution.Repository
	V2MetadataService metadata.V2MetadataService
	tmpFile           *os.File
	verifier          digest.Verifier
	src               distribution.Descriptor
}

func (ld *v2LayerDescriptor) Key() string {
	return "v2:" + ld.digest.String()
}

func (ld *v2LayerDescriptor) ID() string {
	return stringid.TruncateID(ld.digest.String())
}

func (ld *v2LayerDescriptor) DiffID() (layer.DiffID, error) {
	if ld.diffID != "" {
		return ld.diffID, nil
	}
	return ld.V2MetadataService.GetDiffID(ld.digest)
}

func (ld *v2LayerDescriptor) Download(ctx context.Context, progressOutput progress.Output) (io.ReadCloser, int64, error) {
	logrus.Debugf("pulling blob %q", ld.digest)

	var (
		err    error
		offset int64
	)

	if ld.tmpFile == nil {
		ld.tmpFile, err = createDownloadFile()
		if err != nil {
			return nil, 0, xfer.DoNotRetry{Err: err}
		}
	} else {
		offset, err = ld.tmpFile.Seek(0, os.SEEK_END)
		if err != nil {
			logrus.Debugf("error seeking to end of download file: %v", err)
			offset = 0

			ld.tmpFile.Close()
			if err := os.Remove(ld.tmpFile.Name()); err != nil {
				logrus.Errorf("Failed to remove temp file: %s", ld.tmpFile.Name())
			}
			ld.tmpFile, err = createDownloadFile()
			if err != nil {
				return nil, 0, xfer.DoNotRetry{Err: err}
			}
		} else if offset != 0 {
			logrus.Debugf("attempting to resume download of %q from %d bytes", ld.digest, offset)
		}
	}

	tmpFile := ld.tmpFile

	layerDownload, err := ld.open(ctx)
	if err != nil {
		logrus.Errorf("Error initiating layer download: %v", err)
		return nil, 0, retryOnError(err)
	}

	if offset != 0 {
		_, err := layerDownload.Seek(offset, os.SEEK_SET)
		if err != nil {
			if err := ld.truncateDownloadFile(); err != nil {
				return nil, 0, xfer.DoNotRetry{Err: err}
			}
			return nil, 0, err
		}
	}
	size, err := layerDownload.Seek(0, os.SEEK_END)
	if err != nil {
		// Seek failed, perhaps because there was no Content-Length
		// header. This shouldn't fail the download, because we can
		// still continue without a progress bar.
		size = 0
	} else {
		if size != 0 && offset > size {
			logrus.Debug("Partial download is larger than full blob. Starting over")
			offset = 0
			if err := ld.truncateDownloadFile(); err != nil {
				return nil, 0, xfer.DoNotRetry{Err: err}
			}
		}

		// Restore the seek offset either at the beginning of the
		// stream, or just after the last byte we have from previous
		// attempts.
		_, err = layerDownload.Seek(offset, os.SEEK_SET)
		if err != nil {
			return nil, 0, err
		}
	}

	reader := progress.NewProgressReader(ioutils.NewCancelReadCloser(ctx, layerDownload), progressOutput, size-offset, ld.ID(), "Downloading")
	defer reader.Close()

	if ld.verifier == nil {
		ld.verifier = ld.digest.Verifier()
	}

	_, err = io.Copy(tmpFile, io.TeeReader(reader, ld.verifier))
	if err != nil {
		if err == transport.ErrWrongCodeForByteRange {
			if err := ld.truncateDownloadFile(); err != nil {
				return nil, 0, xfer.DoNotRetry{Err: err}
			}
			return nil, 0, err
		}
		return nil, 0, retryOnError(err)
	}

	progress.Update(progressOutput, ld.ID(), "Verifying Checksum")

	if !ld.verifier.Verified() {
		err = fmt.Errorf("filesystem layer verification failed for digest %s", ld.digest)
		logrus.Error(err)

		// Allow a retry if this digest verification error happened
		// after a resumed download.
		if offset != 0 {
			if err := ld.truncateDownloadFile(); err != nil {
				return nil, 0, xfer.DoNotRetry{Err: err}
			}

			return nil, 0, err
		}
		return nil, 0, xfer.DoNotRetry{Err: err}
	}

	progress.Update(progressOutput, ld.ID(), "Download complete")

	logrus.Debugf("Downloaded %s to tempfile %s", ld.ID(), tmpFile.Name())

	_, err = tmpFile.Seek(0, os.SEEK_SET)
	if err != nil {
		tmpFile.Close()
		if err := os.Remove(tmpFile.Name()); err != nil {
			logrus.Errorf("Failed to remove temp file: %s", tmpFile.Name())
		}
		ld.tmpFile = nil
		ld.verifier = nil
		return nil, 0, xfer.DoNotRetry{Err: err}
	}

	// hand off the temporary file to the download manager, so it will only
	// be closed once
	ld.tmpFile = nil

	return ioutils.NewReadCloserWrapper(tmpFile, func() error {
		tmpFile.Close()
		err := os.RemoveAll(tmpFile.Name())
		if err != nil {
			logrus.Errorf("Failed to remove temp file: %s", tmpFile.Name())
		}
		return err
	}), size, nil
}

func (ld *v2LayerDescriptor) Close() {
	if ld.tmpFile != nil {
		ld.tmpFile.Close()
		if err := os.RemoveAll(ld.tmpFile.Name()); err != nil {
			logrus.Errorf("Failed to remove temp file: %s", ld.tmpFile.Name())
		}
	}
}

func (ld *v2LayerDescriptor) truncateDownloadFile() error {
	// Need a new hash context since we will be redoing the download
	ld.verifier = nil

	if _, err := ld.tmpFile.Seek(0, os.SEEK_SET); err != nil {
		logrus.Errorf("error seeking to beginning of download file: %v", err)
		return err
	}

	if err := ld.tmpFile.Truncate(0); err != nil {
		logrus.Errorf("error truncating download file: %v", err)
		return err
	}

	return nil
}

func (ld *v2LayerDescriptor) Registered(diffID layer.DiffID) {
	// Cache mapping from this layer's DiffID to the blobsum
	ld.V2MetadataService.Add(diffID, metadata.V2Metadata{Digest: ld.digest, SourceRepository: ld.repoInfo.Name.Name()})
}

func (p *v2Puller) pullV2Tag(ctx context.Context, ref reference.Named) (tagUpdated bool, err error) {
	manSvc, err := p.repo.Manifests(ctx)
	if err != nil {
		return false, err
	}

	var (
		manifest    distribution.Manifest
		tagOrDigest string // Used for logging/progress only
	)
	if digested, isDigested := ref.(reference.Canonical); isDigested {
		manifest, err = manSvc.Get(ctx, digested.Digest())
		if err != nil {
			return false, err
		}
		tagOrDigest = digested.Digest().String()
	} else if tagged, isTagged := ref.(reference.NamedTagged); isTagged {
		manifest, err = manSvc.Get(ctx, "", distribution.WithTag(tagged.Tag()))
		if err != nil {
			return false, allowV1Fallback(err)
		}
		tagOrDigest = tagged.Tag()
	} else {
		return false, fmt.Errorf("internal error: reference has neither a tag nor a digest: %s", reference.FamiliarString(ref))
	}

	if manifest == nil {
		return false, fmt.Errorf("image manifest does not exist for tag or digest %q", tagOrDigest)
	}

	if m, ok := manifest.(*schema2.DeserializedManifest); ok {
		var allowedMediatype bool
		for _, t := range p.config.Schema2Types {
			if m.Manifest.Config.MediaType == t {
				allowedMediatype = true
				break
			}
		}
		if !allowedMediatype {
			configClass := mediaTypeClasses[m.Manifest.Config.MediaType]
			if configClass == "" {
				configClass = "unknown"
			}
			return false, fmt.Errorf("Encountered remote %q(%s) when fetching", m.Manifest.Config.MediaType, configClass)
		}
	}

	// If manSvc.Get succeeded, we can be confident that the registry on
	// the other side speaks the v2 protocol.
	p.confirmedV2 = true

	logrus.Debugf("Pulling ref from V2 registry: %s", reference.FamiliarString(ref))
	progress.Message(p.config.ProgressOutput, tagOrDigest, "Pulling from "+reference.FamiliarName(p.repo.Named()))

	var (
		id             digest.Digest
		manifestDigest digest.Digest
	)

	switch v := manifest.(type) {
	case *schema1.SignedManifest:
		if p.config.RequireSchema2 {
			return false, fmt.Errorf("invalid manifest: not schema2")
		}
		id, manifestDigest, err = p.pullSchema1(ctx, ref, v)
		if err != nil {
			return false, err
		}
	case *schema2.DeserializedManifest:
		id, manifestDigest, err = p.pullSchema2(ctx, ref, v)
		if err != nil {
			return false, err
		}
	case *manifestlist.DeserializedManifestList:
		id, manifestDigest, err = p.pullManifestList(ctx, ref, v)
		if err != nil {
			return false, err
		}
	default:
		return false, errors.New("unsupported manifest format")
	}

	progress.Message(p.config.ProgressOutput, "", "Digest: "+manifestDigest.String())

	if p.config.ReferenceStore != nil {
		oldTagID, err := p.config.ReferenceStore.Get(ref)
		if err == nil {
			if oldTagID == id {
				return false, addDigestReference(p.config.ReferenceStore, ref, manifestDigest, id)
			}
		} else if err != refstore.ErrDoesNotExist {
			return false, err
		}

		if canonical, ok := ref.(reference.Canonical); ok {
			if err = p.config.ReferenceStore.AddDigest(canonical, id, true); err != nil {
				return false, err
			}
		} else {
			if err = addDigestReference(p.config.ReferenceStore, ref, manifestDigest, id); err != nil {
				return false, err
			}
			if err = p.config.ReferenceStore.AddTag(ref, id, true); err != nil {
				return false, err
			}
		}
	}
	return true, nil
}

func (p *v2Puller) pullSchema1(ctx context.Context, ref reference.Named, unverifiedManifest *schema1.SignedManifest) (id digest.Digest, manifestDigest digest.Digest, err error) {
	var verifiedManifest *schema1.Manifest
	verifiedManifest, err = verifySchema1Manifest(unverifiedManifest, ref)
	if err != nil {
		return "", "", err
	}

	rootFS := image.NewRootFS()

	// remove duplicate layers and check parent chain validity
	err = fixManifestLayers(verifiedManifest)
	if err != nil {
		return "", "", err
	}

	var descriptors []xfer.DownloadDescriptor

	// Image history converted to the new format
	var history []image.History

	// Note that the order of this loop is in the direction of bottom-most
	// to top-most, so that the downloads slice gets ordered correctly.
	for i := len(verifiedManifest.FSLayers) - 1; i >= 0; i-- {
		blobSum := verifiedManifest.FSLayers[i].BlobSum

		var throwAway struct {
			ThrowAway bool `json:"throwaway,omitempty"`
		}
		if err := json.Unmarshal([]byte(verifiedManifest.History[i].V1Compatibility), &throwAway); err != nil {
			return "", "", err
		}

		h, err := v1.HistoryFromConfig([]byte(verifiedManifest.History[i].V1Compatibility), throwAway.ThrowAway)
		if err != nil {
			return "", "", err
		}
		history = append(history, h)

		if throwAway.ThrowAway {
			continue
		}

		layerDescriptor := &v2LayerDescriptor{
			digest:            blobSum,
			repoInfo:          p.repoInfo,
			repo:              p.repo,
			V2MetadataService: p.V2MetadataService,
		}

		descriptors = append(descriptors, layerDescriptor)
	}

	// The v1 manifest itself doesn't directly contain a platform. However,
	// the history does, but unfortunately that's a string, so search through
	// all the history until hopefully we find one which indicates the os.
	platform := runtime.GOOS
	if system.LCOWSupported() {
		type config struct {
			Os string `json:"os,omitempty"`
		}
		for _, v := range verifiedManifest.History {
			var c config
			if err := json.Unmarshal([]byte(v.V1Compatibility), &c); err == nil {
				if c.Os != "" {
					platform = c.Os
					break
				}
			}
		}
	}

	resultRootFS, release, err := p.config.DownloadManager.Download(ctx, *rootFS, layer.Platform(platform), descriptors, p.config.ProgressOutput)
	if err != nil {
		return "", "", err
	}
	defer release()

	config, err := v1.MakeConfigFromV1Config([]byte(verifiedManifest.History[0].V1Compatibility), &resultRootFS, history)
	if err != nil {
		return "", "", err
	}

	imageID, err := p.config.ImageStore.Put(config)
	if err != nil {
		return "", "", err
	}

	manifestDigest = digest.FromBytes(unverifiedManifest.Canonical)

	return imageID, manifestDigest, nil
}

func (p *v2Puller) pullSchema2(ctx context.Context, ref reference.Named, mfst *schema2.DeserializedManifest) (id digest.Digest, manifestDigest digest.Digest, err error) {
	manifestDigest, err = schema2ManifestDigest(ref, mfst)
	if err != nil {
		return "", "", err
	}

	target := mfst.Target()
	if _, err := p.config.ImageStore.Get(target.Digest); err == nil {
		// If the image already exists locally, no need to pull
		// anything.
		return target.Digest, manifestDigest, nil
	}

	var descriptors []xfer.DownloadDescriptor

	// Note that the order of this loop is in the direction of bottom-most
	// to top-most, so that the downloads slice gets ordered correctly.
	for _, d := range mfst.Layers {
		layerDescriptor := &v2LayerDescriptor{
			digest:            d.Digest,
			repo:              p.repo,
			repoInfo:          p.repoInfo,
			V2MetadataService: p.V2MetadataService,
			src:               d,
		}

		descriptors = append(descriptors, layerDescriptor)
	}

	configChan := make(chan []byte, 1)
	configErrChan := make(chan error, 1)
	layerErrChan := make(chan error, 1)
	downloadsDone := make(chan struct{})
	var cancel func()
	ctx, cancel = context.WithCancel(ctx)
	defer cancel()

	// Pull the image config
	go func() {
		configJSON, err := p.pullSchema2Config(ctx, target.Digest)
		if err != nil {
			configErrChan <- ImageConfigPullError{Err: err}
			cancel()
			return
		}
		configChan <- configJSON
	}()

	var (
		configJSON       []byte         // raw serialized image config
		downloadedRootFS *image.RootFS  // rootFS from registered layers
		configRootFS     *image.RootFS  // rootFS from configuration
		release          func()         // release resources from rootFS download
		platform         layer.Platform // for LCOW when registering downloaded layers
	)

	// https://github.com/docker/docker/issues/24766 - Err on the side of caution,
	// explicitly blocking images intended for linux from the Windows daemon. On
	// Windows, we do this before the attempt to download, effectively serialising
	// the download slightly slowing it down. We have to do it this way, as
	// chances are the download of layers itself would fail due to file names
	// which aren't suitable for NTFS. At some point in the future, if a similar
	// check to block Windows images being pulled on Linux is implemented, it
	// may be necessary to perform the same type of serialisation.
	if runtime.GOOS == "windows" {
		configJSON, configRootFS, platform, err = receiveConfig(p.config.ImageStore, configChan, configErrChan)
		if err != nil {
			return "", "", err
		}

		if configRootFS == nil {
			return "", "", errRootFSInvalid
		}

		if len(descriptors) != len(configRootFS.DiffIDs) {
			return "", "", errRootFSMismatch
		}

		// Populate diff ids in descriptors to avoid downloading foreign layers
		// which have been side loaded
		for i := range descriptors {
			descriptors[i].(*v2LayerDescriptor).diffID = configRootFS.DiffIDs[i]
		}
	}

	if p.config.DownloadManager != nil {
		go func() {
			var (
				err    error
				rootFS image.RootFS
			)
			downloadRootFS := *image.NewRootFS()
			rootFS, release, err = p.config.DownloadManager.Download(ctx, downloadRootFS, platform, descriptors, p.config.ProgressOutput)
			if err != nil {
				// Intentionally do not cancel the config download here
				// as the error from config download (if there is one)
				// is more interesting than the layer download error
				layerErrChan <- err
				return
			}

			downloadedRootFS = &rootFS
			close(downloadsDone)
		}()
	} else {
		// We have nothing to download
		close(downloadsDone)
	}

	if configJSON == nil {
		configJSON, configRootFS, _, err = receiveConfig(p.config.ImageStore, configChan, configErrChan)
		if err == nil && configRootFS == nil {
			err = errRootFSInvalid
		}
		if err != nil {
			cancel()
			select {
			case <-downloadsDone:
			case <-layerErrChan:
			}
			return "", "", err
		}
	}

	select {
	case <-downloadsDone:
	case err = <-layerErrChan:
		return "", "", err
	}

	if release != nil {
		defer release()
	}

	if downloadedRootFS != nil {
		// The DiffIDs returned in rootFS MUST match those in the config.
		// Otherwise the image config could be referencing layers that aren't
		// included in the manifest.
		if len(downloadedRootFS.DiffIDs) != len(configRootFS.DiffIDs) {
			return "", "", errRootFSMismatch
		}

		for i := range downloadedRootFS.DiffIDs {
			if downloadedRootFS.DiffIDs[i] != configRootFS.DiffIDs[i] {
				return "", "", errRootFSMismatch
			}
		}
	}

	imageID, err := p.config.ImageStore.Put(configJSON)
	if err != nil {
		return "", "", err
	}

	return imageID, manifestDigest, nil
}

func receiveConfig(s ImageConfigStore, configChan <-chan []byte, errChan <-chan error) ([]byte, *image.RootFS, layer.Platform, error) {
	select {
	case configJSON := <-configChan:
		rootfs, platform, err := s.RootFSAndPlatformFromConfig(configJSON)
		if err != nil {
			return nil, nil, "", err
		}
		return configJSON, rootfs, platform, nil
	case err := <-errChan:
		return nil, nil, "", err
		// Don't need a case for ctx.Done in the select because cancellation
		// will trigger an error in p.pullSchema2ImageConfig.
	}
}

// pullManifestList handles "manifest lists" which point to various
// platform-specific manifests.
func (p *v2Puller) pullManifestList(ctx context.Context, ref reference.Named, mfstList *manifestlist.DeserializedManifestList) (id digest.Digest, manifestListDigest digest.Digest, err error) {
	manifestListDigest, err = schema2ManifestDigest(ref, mfstList)
	if err != nil {
		return "", "", err
	}

	logrus.Debugf("%s resolved to a manifestList object with %d entries; looking for a os/arch match", ref, len(mfstList.Manifests))
	var manifestDigest digest.Digest
	for _, manifestDescriptor := range mfstList.Manifests {
		// TODO(aaronl): The manifest list spec supports optional
		// "features" and "variant" fields. These are not yet used.
		// Once they are, their values should be interpreted here.
		if manifestDescriptor.Platform.Architecture == runtime.GOARCH && manifestDescriptor.Platform.OS == runtime.GOOS {
			manifestDigest = manifestDescriptor.Digest
			logrus.Debugf("found match for %s/%s with media type %s, digest %s", runtime.GOOS, runtime.GOARCH, manifestDescriptor.MediaType, manifestDigest.String())
			break
		}
	}

	if manifestDigest == "" {
		errMsg := fmt.Sprintf("no matching manifest for %s/%s in the manifest list entries", runtime.GOOS, runtime.GOARCH)
		logrus.Debugf(errMsg)
		return "", "", errors.New(errMsg)
	}

	manSvc, err := p.repo.Manifests(ctx)
	if err != nil {
		return "", "", err
	}

	manifest, err := manSvc.Get(ctx, manifestDigest)
	if err != nil {
		return "", "", err
	}

	manifestRef, err := reference.WithDigest(reference.TrimNamed(ref), manifestDigest)
	if err != nil {
		return "", "", err
	}

	switch v := manifest.(type) {
	case *schema1.SignedManifest:
		id, _, err = p.pullSchema1(ctx, manifestRef, v)
		if err != nil {
			return "", "", err
		}
	case *schema2.DeserializedManifest:
		id, _, err = p.pullSchema2(ctx, manifestRef, v)
		if err != nil {
			return "", "", err
		}
	default:
		return "", "", errors.New("unsupported manifest format")
	}

	return id, manifestListDigest, err
}

func (p *v2Puller) pullSchema2Config(ctx context.Context, dgst digest.Digest) (configJSON []byte, err error) {
	blobs := p.repo.Blobs(ctx)
	configJSON, err = blobs.Get(ctx, dgst)
	if err != nil {
		return nil, err
	}

	// Verify image config digest
	verifier := dgst.Verifier()
	if _, err := verifier.Write(configJSON); err != nil {
		return nil, err
	}
	if !verifier.Verified() {
		err := fmt.Errorf("image config verification failed for digest %s", dgst)
		logrus.Error(err)
		return nil, err
	}

	return configJSON, nil
}

// schema2ManifestDigest computes the manifest digest, and, if pulling by
// digest, ensures that it matches the requested digest.
func schema2ManifestDigest(ref reference.Named, mfst distribution.Manifest) (digest.Digest, error) {
	_, canonical, err := mfst.Payload()
	if err != nil {
		return "", err
	}

	// If pull by digest, then verify the manifest digest.
	if digested, isDigested := ref.(reference.Canonical); isDigested {
		verifier := digested.Digest().Verifier()
		if _, err := verifier.Write(canonical); err != nil {
			return "", err
		}
		if !verifier.Verified() {
			err := fmt.Errorf("manifest verification failed for digest %s", digested.Digest())
			logrus.Error(err)
			return "", err
		}
		return digested.Digest(), nil
	}

	return digest.FromBytes(canonical), nil
}

// allowV1Fallback checks if the error is a possible reason to fallback to v1
// (even if confirmedV2 has been set already), and if so, wraps the error in
// a fallbackError with confirmedV2 set to false. Otherwise, it returns the
// error unmodified.
func allowV1Fallback(err error) error {
	switch v := err.(type) {
	case errcode.Errors:
		if len(v) != 0 {
			if v0, ok := v[0].(errcode.Error); ok && shouldV2Fallback(v0) {
				return fallbackError{
					err:         err,
					confirmedV2: false,
					transportOK: true,
				}
			}
		}
	case errcode.Error:
		if shouldV2Fallback(v) {
			return fallbackError{
				err:         err,
				confirmedV2: false,
				transportOK: true,
			}
		}
	case *url.Error:
		if v.Err == auth.ErrNoBasicAuthCredentials {
			return fallbackError{err: err, confirmedV2: false}
		}
	}

	return err
}

func verifySchema1Manifest(signedManifest *schema1.SignedManifest, ref reference.Named) (m *schema1.Manifest, err error) {
	// If pull by digest, then verify the manifest digest. NOTE: It is
	// important to do this first, before any other content validation. If the
	// digest cannot be verified, don't even bother with those other things.
	if digested, isCanonical := ref.(reference.Canonical); isCanonical {
		verifier := digested.Digest().Verifier()
		if _, err := verifier.Write(signedManifest.Canonical); err != nil {
			return nil, err
		}
		if !verifier.Verified() {
			err := fmt.Errorf("image verification failed for digest %s", digested.Digest())
			logrus.Error(err)
			return nil, err
		}
	}
	m = &signedManifest.Manifest

	if m.SchemaVersion != 1 {
		return nil, fmt.Errorf("unsupported schema version %d for %q", m.SchemaVersion, reference.FamiliarString(ref))
	}
	if len(m.FSLayers) != len(m.History) {
		return nil, fmt.Errorf("length of history not equal to number of layers for %q", reference.FamiliarString(ref))
	}
	if len(m.FSLayers) == 0 {
		return nil, fmt.Errorf("no FSLayers in manifest for %q", reference.FamiliarString(ref))
	}
	return m, nil
}

// fixManifestLayers removes repeated layers from the manifest and checks the
// correctness of the parent chain.
func fixManifestLayers(m *schema1.Manifest) error {
	imgs := make([]*image.V1Image, len(m.FSLayers))
	for i := range m.FSLayers {
		img := &image.V1Image{}

		if err := json.Unmarshal([]byte(m.History[i].V1Compatibility), img); err != nil {
			return err
		}

		imgs[i] = img
		if err := v1.ValidateID(img.ID); err != nil {
			return err
		}
	}

	if imgs[len(imgs)-1].Parent != "" && runtime.GOOS != "windows" {
		// Windows base layer can point to a base layer parent that is not in manifest.
		return errors.New("invalid parent ID in the base layer of the image")
	}

	// check general duplicates to error instead of a deadlock
	idmap := make(map[string]struct{})

	var lastID string
	for _, img := range imgs {
		// skip IDs that appear after each other, we handle those later
		if _, exists := idmap[img.ID]; img.ID != lastID && exists {
			return fmt.Errorf("ID %+v appears multiple times in manifest", img.ID)
		}
		lastID = img.ID
		idmap[lastID] = struct{}{}
	}

	// backwards loop so that we keep the remaining indexes after removing items
	for i := len(imgs) - 2; i >= 0; i-- {
		if imgs[i].ID == imgs[i+1].ID { // repeated ID. remove and continue
			m.FSLayers = append(m.FSLayers[:i], m.FSLayers[i+1:]...)
			m.History = append(m.History[:i], m.History[i+1:]...)
		} else if imgs[i].Parent != imgs[i+1].ID {
			return fmt.Errorf("Invalid parent ID. Expected %v, got %v.", imgs[i+1].ID, imgs[i].Parent)
		}
	}

	return nil
}

func createDownloadFile() (*os.File, error) {
	return ioutil.TempFile("", "GetImageBlob")
}
