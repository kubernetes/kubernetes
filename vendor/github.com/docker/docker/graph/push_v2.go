package graph

import (
	"fmt"
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
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/utils"
)

type v2Pusher struct {
	*TagStore
	endpoint  registry.APIEndpoint
	localRepo Repository
	repoInfo  *registry.RepositoryInfo
	config    *ImagePushConfig
	sf        *streamformatter.StreamFormatter
	repo      distribution.Repository
}

func (p *v2Pusher) Push() (fallback bool, err error) {
	p.repo, err = NewV2Repository(p.repoInfo, p.endpoint, p.config.MetaHeaders, p.config.AuthConfig)
	if err != nil {
		logrus.Debugf("Error getting v2 registry: %v", err)
		return true, err
	}
	return false, p.pushV2Repository(p.config.Tag)
}

func (p *v2Pusher) getImageTags(askedTag string) ([]string, error) {
	logrus.Debugf("Checking %q against %#v", askedTag, p.localRepo)
	if len(askedTag) > 0 {
		if _, ok := p.localRepo[askedTag]; !ok || utils.DigestReference(askedTag) {
			return nil, fmt.Errorf("Tag does not exist for %s", askedTag)
		}
		return []string{askedTag}, nil
	}
	var tags []string
	for tag := range p.localRepo {
		if !utils.DigestReference(tag) {
			tags = append(tags, tag)
		}
	}
	return tags, nil
}

func (p *v2Pusher) pushV2Repository(tag string) error {
	localName := p.repoInfo.LocalName
	if _, err := p.poolAdd("push", localName); err != nil {
		return err
	}
	defer p.poolRemove("push", localName)

	tags, err := p.getImageTags(tag)
	if err != nil {
		return fmt.Errorf("error getting tags for %s: %s", localName, err)
	}
	if len(tags) == 0 {
		return fmt.Errorf("no tags to push for %s", localName)
	}

	for _, tag := range tags {
		if err := p.pushV2Tag(tag); err != nil {
			return err
		}
	}

	return nil
}

func (p *v2Pusher) pushV2Tag(tag string) error {
	logrus.Debugf("Pushing repository: %s:%s", p.repo.Name(), tag)

	layerId, exists := p.localRepo[tag]
	if !exists {
		return fmt.Errorf("tag does not exist: %s", tag)
	}

	layersSeen := make(map[string]bool)

	layer, err := p.graph.Get(layerId)
	if err != nil {
		return err
	}

	m := &manifest.Manifest{
		Versioned: manifest.Versioned{
			SchemaVersion: 1,
		},
		Name:         p.repo.Name(),
		Tag:          tag,
		Architecture: layer.Architecture,
		FSLayers:     []manifest.FSLayer{},
		History:      []manifest.History{},
	}

	var metadata runconfig.Config
	if layer != nil && layer.Config != nil {
		metadata = *layer.Config
	}

	out := p.config.OutStream

	for ; layer != nil; layer, err = p.graph.GetParent(layer) {
		if err != nil {
			return err
		}

		if layersSeen[layer.ID] {
			break
		}

		logrus.Debugf("Pushing layer: %s", layer.ID)

		if layer.Config != nil && metadata.Image != layer.ID {
			if err := runconfig.Merge(&metadata, layer.Config); err != nil {
				return err
			}
		}

		jsonData, err := p.graph.RawJSON(layer.ID)
		if err != nil {
			return fmt.Errorf("cannot retrieve the path for %s: %s", layer.ID, err)
		}

		var exists bool
		dgst, err := p.graph.GetDigest(layer.ID)
		switch err {
		case nil:
			_, err := p.repo.Blobs(nil).Stat(nil, dgst)
			switch err {
			case nil:
				exists = true
				out.Write(p.sf.FormatProgress(stringid.TruncateID(layer.ID), "Image already exists", nil))
			case distribution.ErrBlobUnknown:
				// nop
			default:
				out.Write(p.sf.FormatProgress(stringid.TruncateID(layer.ID), "Image push failed", nil))
				return err
			}
		case ErrDigestNotSet:
			// nop
		case digest.ErrDigestInvalidFormat, digest.ErrDigestUnsupported:
			return fmt.Errorf("error getting image checksum: %v", err)
		}

		// if digest was empty or not saved, or if blob does not exist on the remote repository,
		// then fetch it.
		if !exists {
			if pushDigest, err := p.pushV2Image(p.repo.Blobs(nil), layer); err != nil {
				return err
			} else if pushDigest != dgst {
				// Cache new checksum
				if err := p.graph.SetDigest(layer.ID, pushDigest); err != nil {
					return err
				}
				dgst = pushDigest
			}
		}

		m.FSLayers = append(m.FSLayers, manifest.FSLayer{BlobSum: dgst})
		m.History = append(m.History, manifest.History{V1Compatibility: string(jsonData)})

		layersSeen[layer.ID] = true
	}

	logrus.Infof("Signed manifest for %s:%s using daemon's key: %s", p.repo.Name(), tag, p.trustKey.KeyID())
	signed, err := manifest.Sign(m, p.trustKey)
	if err != nil {
		return err
	}

	manifestDigest, err := digestFromManifest(signed, p.repo.Name())
	if err != nil {
		return err
	}
	if manifestDigest != "" {
		out.Write(p.sf.FormatStatus("", "Digest: %s", manifestDigest))
	}

	return p.repo.Manifests().Put(signed)
}

func (p *v2Pusher) pushV2Image(bs distribution.BlobService, img *image.Image) (digest.Digest, error) {
	out := p.config.OutStream

	out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), "Buffering to Disk", nil))

	image, err := p.graph.Get(img.ID)
	if err != nil {
		return "", err
	}
	arch, err := p.graph.TarLayer(image)
	if err != nil {
		return "", err
	}

	tf, err := p.graph.newTempFile()
	if err != nil {
		return "", err
	}
	defer func() {
		tf.Close()
		os.Remove(tf.Name())
	}()

	size, dgst, err := bufferToFile(tf, arch)
	if err != nil {
		return "", err
	}

	// Send the layer
	logrus.Debugf("rendered layer for %s of [%d] size", img.ID, size)
	layerUpload, err := bs.Create(nil)
	if err != nil {
		return "", err
	}
	defer layerUpload.Close()

	reader := progressreader.New(progressreader.Config{
		In:        ioutil.NopCloser(tf),
		Out:       out,
		Formatter: p.sf,
		Size:      int(size),
		NewLines:  false,
		ID:        stringid.TruncateID(img.ID),
		Action:    "Pushing",
	})
	n, err := layerUpload.ReadFrom(reader)
	if err != nil {
		return "", err
	}
	if n != size {
		return "", fmt.Errorf("short upload: only wrote %d of %d", n, size)
	}

	desc := distribution.Descriptor{Digest: dgst}
	if _, err := layerUpload.Commit(nil, desc); err != nil {
		return "", err
	}

	out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), "Image successfully pushed", nil))

	return dgst, nil
}
