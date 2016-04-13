package graph

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/Sirupsen/logrus"
	"github.com/docker/distribution"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/manifest"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/progressreader"
	"github.com/docker/docker/pkg/streamformatter"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/trust"
	"github.com/docker/docker/utils"
	"github.com/docker/libtrust"
)

type v2Puller struct {
	*TagStore
	endpoint  registry.APIEndpoint
	config    *ImagePullConfig
	sf        *streamformatter.StreamFormatter
	repoInfo  *registry.RepositoryInfo
	repo      distribution.Repository
	sessionID string
}

func (p *v2Puller) Pull(tag string) (fallback bool, err error) {
	// TODO(tiborvass): was ReceiveTimeout
	p.repo, err = NewV2Repository(p.repoInfo, p.endpoint, p.config.MetaHeaders, p.config.AuthConfig)
	if err != nil {
		logrus.Debugf("Error getting v2 registry: %v", err)
		return true, err
	}

	p.sessionID = stringid.GenerateRandomID()

	if err := p.pullV2Repository(tag); err != nil {
		if registry.ContinueOnError(err) {
			logrus.Debugf("Error trying v2 registry: %v", err)
			return true, err
		}
		return false, err
	}
	return false, nil
}

func (p *v2Puller) pullV2Repository(tag string) (err error) {
	var tags []string
	taggedName := p.repoInfo.LocalName
	if len(tag) > 0 {
		tags = []string{tag}
		taggedName = utils.ImageReference(p.repoInfo.LocalName, tag)
	} else {
		var err error
		tags, err = p.repo.Manifests().Tags()
		if err != nil {
			return err
		}

	}

	c, err := p.poolAdd("pull", taggedName)
	if err != nil {
		if c != nil {
			// Another pull of the same repository is already taking place; just wait for it to finish
			p.sf.FormatStatus("", "Repository %s already being pulled by another client. Waiting.", p.repoInfo.CanonicalName)
			<-c
			return nil
		}
		return err
	}
	defer p.poolRemove("pull", taggedName)

	var layersDownloaded bool
	for _, tag := range tags {
		// pulledNew is true if either new layers were downloaded OR if existing images were newly tagged
		// TODO(tiborvass): should we change the name of `layersDownload`? What about message in WriteStatus?
		pulledNew, err := p.pullV2Tag(tag, taggedName)
		if err != nil {
			return err
		}
		layersDownloaded = layersDownloaded || pulledNew
	}

	WriteStatus(taggedName, p.config.OutStream, p.sf, layersDownloaded)

	return nil
}

// downloadInfo is used to pass information from download to extractor
type downloadInfo struct {
	img      *image.Image
	tmpFile  *os.File
	digest   digest.Digest
	layer    distribution.ReadSeekCloser
	size     int64
	err      chan error
	verified bool
}

type errVerification struct{}

func (errVerification) Error() string { return "verification failed" }

func (p *v2Puller) download(di *downloadInfo) {
	logrus.Debugf("pulling blob %q to %s", di.digest, di.img.ID)

	out := p.config.OutStream

	if c, err := p.poolAdd("pull", "img:"+di.img.ID); err != nil {
		if c != nil {
			out.Write(p.sf.FormatProgress(stringid.TruncateID(di.img.ID), "Layer already being pulled by another client. Waiting.", nil))
			<-c
			out.Write(p.sf.FormatProgress(stringid.TruncateID(di.img.ID), "Download complete", nil))
		} else {
			logrus.Debugf("Image (id: %s) pull is already running, skipping: %v", di.img.ID, err)
		}
		di.err <- nil
		return
	}

	defer p.poolRemove("pull", "img:"+di.img.ID)
	tmpFile, err := ioutil.TempFile("", "GetImageBlob")
	if err != nil {
		di.err <- err
		return
	}

	blobs := p.repo.Blobs(nil)

	desc, err := blobs.Stat(nil, di.digest)
	if err != nil {
		logrus.Debugf("Error statting layer: %v", err)
		di.err <- err
		return
	}
	di.size = desc.Length

	layerDownload, err := blobs.Open(nil, di.digest)
	if err != nil {
		logrus.Debugf("Error fetching layer: %v", err)
		di.err <- err
		return
	}
	defer layerDownload.Close()

	verifier, err := digest.NewDigestVerifier(di.digest)
	if err != nil {
		di.err <- err
		return
	}

	reader := progressreader.New(progressreader.Config{
		In:        ioutil.NopCloser(io.TeeReader(layerDownload, verifier)),
		Out:       out,
		Formatter: p.sf,
		Size:      int(di.size),
		NewLines:  false,
		ID:        stringid.TruncateID(di.img.ID),
		Action:    "Downloading",
	})
	io.Copy(tmpFile, reader)

	out.Write(p.sf.FormatProgress(stringid.TruncateID(di.img.ID), "Verifying Checksum", nil))

	di.verified = verifier.Verified()
	if !di.verified {
		logrus.Infof("Image verification failed for layer %s", di.digest)
	}

	out.Write(p.sf.FormatProgress(stringid.TruncateID(di.img.ID), "Download complete", nil))

	logrus.Debugf("Downloaded %s to tempfile %s", di.img.ID, tmpFile.Name())
	di.tmpFile = tmpFile
	di.layer = layerDownload

	di.err <- nil
}

func (p *v2Puller) pullV2Tag(tag, taggedName string) (bool, error) {
	logrus.Debugf("Pulling tag from V2 registry: %q", tag)
	out := p.config.OutStream

	manifest, err := p.repo.Manifests().GetByTag(tag)
	if err != nil {
		return false, err
	}
	verified, err := p.validateManifest(manifest, tag)
	if err != nil {
		return false, err
	}
	if verified {
		logrus.Printf("Image manifest for %s has been verified", taggedName)
	}

	out.Write(p.sf.FormatStatus(tag, "Pulling from %s", p.repo.Name()))

	downloads := make([]downloadInfo, len(manifest.FSLayers))

	layerIDs := []string{}
	defer func() {
		p.graph.Release(p.sessionID, layerIDs...)
	}()

	for i := len(manifest.FSLayers) - 1; i >= 0; i-- {
		img, err := image.NewImgJSON([]byte(manifest.History[i].V1Compatibility))
		if err != nil {
			logrus.Debugf("error getting image v1 json: %v", err)
			return false, err
		}
		downloads[i].img = img
		downloads[i].digest = manifest.FSLayers[i].BlobSum

		p.graph.Retain(p.sessionID, img.ID)
		layerIDs = append(layerIDs, img.ID)

		// Check if exists
		if p.graph.Exists(img.ID) {
			logrus.Debugf("Image already exists: %s", img.ID)
			continue
		}

		out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), "Pulling fs layer", nil))

		downloads[i].err = make(chan error)
		go p.download(&downloads[i])
	}

	var tagUpdated bool
	for i := len(downloads) - 1; i >= 0; i-- {
		d := &downloads[i]
		if d.err != nil {
			if err := <-d.err; err != nil {
				return false, err
			}
		}
		verified = verified && d.verified
		if d.layer != nil {
			// if tmpFile is empty assume download and extracted elsewhere
			defer os.Remove(d.tmpFile.Name())
			defer d.tmpFile.Close()
			d.tmpFile.Seek(0, 0)
			if d.tmpFile != nil {

				reader := progressreader.New(progressreader.Config{
					In:        d.tmpFile,
					Out:       out,
					Formatter: p.sf,
					Size:      int(d.size),
					NewLines:  false,
					ID:        stringid.TruncateID(d.img.ID),
					Action:    "Extracting",
				})

				err = p.graph.Register(d.img, reader)
				if err != nil {
					return false, err
				}

				if err := p.graph.SetDigest(d.img.ID, d.digest); err != nil {
					return false, err
				}

				// FIXME: Pool release here for parallel tag pull (ensures any downloads block until fully extracted)
			}
			out.Write(p.sf.FormatProgress(stringid.TruncateID(d.img.ID), "Pull complete", nil))
			tagUpdated = true
		} else {
			out.Write(p.sf.FormatProgress(stringid.TruncateID(d.img.ID), "Already exists", nil))
		}
	}

	manifestDigest, err := digestFromManifest(manifest, p.repoInfo.LocalName)
	if err != nil {
		return false, err
	}

	// Check for new tag if no layers downloaded
	if !tagUpdated {
		repo, err := p.Get(p.repoInfo.LocalName)
		if err != nil {
			return false, err
		}
		if repo != nil {
			if _, exists := repo[tag]; !exists {
				tagUpdated = true
			}
		} else {
			tagUpdated = true
		}
	}

	if verified && tagUpdated {
		out.Write(p.sf.FormatStatus(p.repo.Name()+":"+tag, "The image you are pulling has been verified. Important: image verification is a tech preview feature and should not be relied on to provide security."))
	}

	if utils.DigestReference(tag) {
		// TODO(stevvooe): Ideally, we should always set the digest so we can
		// use the digest whether we pull by it or not. Unfortunately, the tag
		// store treats the digest as a separate tag, meaning there may be an
		// untagged digest image that would seem to be dangling by a user.
		if err = p.SetDigest(p.repoInfo.LocalName, tag, downloads[0].img.ID); err != nil {
			return false, err
		}
	} else {
		// only set the repository/tag -> image ID mapping when pulling by tag (i.e. not by digest)
		if err = p.Tag(p.repoInfo.LocalName, tag, downloads[0].img.ID, true); err != nil {
			return false, err
		}
	}

	if manifestDigest != "" {
		out.Write(p.sf.FormatStatus("", "Digest: %s", manifestDigest))
	}

	return tagUpdated, nil
}

// verifyTrustedKeys checks the keys provided against the trust store,
// ensuring that the provided keys are trusted for the namespace. The keys
// provided from this method must come from the signatures provided as part of
// the manifest JWS package, obtained from unpackSignedManifest or libtrust.
func (p *v2Puller) verifyTrustedKeys(namespace string, keys []libtrust.PublicKey) (verified bool, err error) {
	if namespace[0] != '/' {
		namespace = "/" + namespace
	}

	for _, key := range keys {
		b, err := key.MarshalJSON()
		if err != nil {
			return false, fmt.Errorf("error marshalling public key: %s", err)
		}
		// Check key has read/write permission (0x03)
		v, err := p.trustService.CheckKey(namespace, b, 0x03)
		if err != nil {
			vErr, ok := err.(trust.NotVerifiedError)
			if !ok {
				return false, fmt.Errorf("error running key check: %s", err)
			}
			logrus.Debugf("Key check result: %v", vErr)
		}
		verified = v
	}

	if verified {
		logrus.Debug("Key check result: verified")
	}

	return
}

func (p *v2Puller) validateManifest(m *manifest.SignedManifest, tag string) (verified bool, err error) {
	// TODO(tiborvass): what's the usecase for having manifest == nil and err == nil ? Shouldn't be the error be "DoesNotExist" ?
	if m == nil {
		return false, fmt.Errorf("image manifest does not exist for tag %q", tag)
	}
	if m.SchemaVersion != 1 {
		return false, fmt.Errorf("unsupported schema version %d for tag %q", m.SchemaVersion, tag)
	}
	if len(m.FSLayers) != len(m.History) {
		return false, fmt.Errorf("length of history not equal to number of layers for tag %q", tag)
	}
	if len(m.FSLayers) == 0 {
		return false, fmt.Errorf("no FSLayers in manifest for tag %q", tag)
	}
	keys, err := manifest.Verify(m)
	if err != nil {
		return false, fmt.Errorf("error verifying manifest for tag %q: %v", tag, err)
	}
	verified, err = p.verifyTrustedKeys(m.Name, keys)
	if err != nil {
		return false, fmt.Errorf("error verifying manifest keys: %v", err)
	}
	localDigest, err := digest.ParseDigest(tag)
	// if pull by digest, then verify
	if err == nil {
		verifier, err := digest.NewDigestVerifier(localDigest)
		if err != nil {
			return false, err
		}
		payload, err := m.Payload()
		if err != nil {
			return false, err
		}
		if _, err := verifier.Write(payload); err != nil {
			return false, err
		}
		verified = verified && verifier.Verified()
	}
	return verified, nil
}
