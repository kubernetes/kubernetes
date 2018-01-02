package distribution

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/distribution/xfer"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/image"
	"github.com/docker/docker/image/v1"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/registry"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

type v1Puller struct {
	v1IDService *metadata.V1IDService
	endpoint    registry.APIEndpoint
	config      *ImagePullConfig
	repoInfo    *registry.RepositoryInfo
	session     *registry.Session
}

func (p *v1Puller) Pull(ctx context.Context, ref reference.Named) error {
	if _, isCanonical := ref.(reference.Canonical); isCanonical {
		// Allowing fallback, because HTTPS v1 is before HTTP v2
		return fallbackError{err: ErrNoSupport{Err: errors.New("Cannot pull by digest with v1 registry")}}
	}

	tlsConfig, err := p.config.RegistryService.TLSConfig(p.repoInfo.Index.Name)
	if err != nil {
		return err
	}
	// Adds Docker-specific headers as well as user-specified headers (metaHeaders)
	tr := transport.NewTransport(
		// TODO(tiborvass): was ReceiveTimeout
		registry.NewTransport(tlsConfig),
		registry.DockerHeaders(dockerversion.DockerUserAgent(ctx), p.config.MetaHeaders)...,
	)
	client := registry.HTTPClient(tr)
	v1Endpoint, err := p.endpoint.ToV1Endpoint(dockerversion.DockerUserAgent(ctx), p.config.MetaHeaders)
	if err != nil {
		logrus.Debugf("Could not get v1 endpoint: %v", err)
		return fallbackError{err: err}
	}
	p.session, err = registry.NewSession(client, p.config.AuthConfig, v1Endpoint)
	if err != nil {
		// TODO(dmcgowan): Check if should fallback
		logrus.Debugf("Fallback from error: %s", err)
		return fallbackError{err: err}
	}
	if err := p.pullRepository(ctx, ref); err != nil {
		// TODO(dmcgowan): Check if should fallback
		return err
	}
	progress.Message(p.config.ProgressOutput, "", p.repoInfo.Name.Name()+": this image was pulled from a legacy registry.  Important: This registry version will not be supported in future versions of docker.")

	return nil
}

func (p *v1Puller) pullRepository(ctx context.Context, ref reference.Named) error {
	progress.Message(p.config.ProgressOutput, "", "Pulling repository "+p.repoInfo.Name.Name())

	tagged, isTagged := ref.(reference.NamedTagged)

	repoData, err := p.session.GetRepositoryData(p.repoInfo.Name)
	if err != nil {
		if strings.Contains(err.Error(), "HTTP code: 404") {
			if isTagged {
				return fmt.Errorf("Error: image %s:%s not found", reference.Path(p.repoInfo.Name), tagged.Tag())
			}
			return fmt.Errorf("Error: image %s not found", reference.Path(p.repoInfo.Name))
		}
		// Unexpected HTTP error
		return err
	}

	logrus.Debug("Retrieving the tag list")
	var tagsList map[string]string
	if !isTagged {
		tagsList, err = p.session.GetRemoteTags(repoData.Endpoints, p.repoInfo.Name)
	} else {
		var tagID string
		tagsList = make(map[string]string)
		tagID, err = p.session.GetRemoteTag(repoData.Endpoints, p.repoInfo.Name, tagged.Tag())
		if err == registry.ErrRepoNotFound {
			return fmt.Errorf("Tag %s not found in repository %s", tagged.Tag(), p.repoInfo.Name.Name())
		}
		tagsList[tagged.Tag()] = tagID
	}
	if err != nil {
		logrus.Errorf("unable to get remote tags: %s", err)
		return err
	}

	for tag, id := range tagsList {
		repoData.ImgList[id] = &registry.ImgData{
			ID:       id,
			Tag:      tag,
			Checksum: "",
		}
	}

	layersDownloaded := false
	for _, imgData := range repoData.ImgList {
		if isTagged && imgData.Tag != tagged.Tag() {
			continue
		}

		err := p.downloadImage(ctx, repoData, imgData, &layersDownloaded)
		if err != nil {
			return err
		}
	}

	writeStatus(reference.FamiliarString(ref), p.config.ProgressOutput, layersDownloaded)
	return nil
}

func (p *v1Puller) downloadImage(ctx context.Context, repoData *registry.RepositoryData, img *registry.ImgData, layersDownloaded *bool) error {
	if img.Tag == "" {
		logrus.Debugf("Image (id: %s) present in this repository but untagged, skipping", img.ID)
		return nil
	}

	localNameRef, err := reference.WithTag(p.repoInfo.Name, img.Tag)
	if err != nil {
		retErr := fmt.Errorf("Image (id: %s) has invalid tag: %s", img.ID, img.Tag)
		logrus.Debug(retErr.Error())
		return retErr
	}

	if err := v1.ValidateID(img.ID); err != nil {
		return err
	}

	progress.Updatef(p.config.ProgressOutput, stringid.TruncateID(img.ID), "Pulling image (%s) from %s", img.Tag, p.repoInfo.Name.Name())
	success := false
	var lastErr error
	for _, ep := range p.repoInfo.Index.Mirrors {
		ep += "v1/"
		progress.Updatef(p.config.ProgressOutput, stringid.TruncateID(img.ID), fmt.Sprintf("Pulling image (%s) from %s, mirror: %s", img.Tag, p.repoInfo.Name.Name(), ep))
		if err = p.pullImage(ctx, img.ID, ep, localNameRef, layersDownloaded); err != nil {
			// Don't report errors when pulling from mirrors.
			logrus.Debugf("Error pulling image (%s) from %s, mirror: %s, %s", img.Tag, p.repoInfo.Name.Name(), ep, err)
			continue
		}
		success = true
		break
	}
	if !success {
		for _, ep := range repoData.Endpoints {
			progress.Updatef(p.config.ProgressOutput, stringid.TruncateID(img.ID), "Pulling image (%s) from %s, endpoint: %s", img.Tag, p.repoInfo.Name.Name(), ep)
			if err = p.pullImage(ctx, img.ID, ep, localNameRef, layersDownloaded); err != nil {
				// It's not ideal that only the last error is returned, it would be better to concatenate the errors.
				// As the error is also given to the output stream the user will see the error.
				lastErr = err
				progress.Updatef(p.config.ProgressOutput, stringid.TruncateID(img.ID), "Error pulling image (%s) from %s, endpoint: %s, %s", img.Tag, p.repoInfo.Name.Name(), ep, err)
				continue
			}
			success = true
			break
		}
	}
	if !success {
		err := fmt.Errorf("Error pulling image (%s) from %s, %v", img.Tag, p.repoInfo.Name.Name(), lastErr)
		progress.Update(p.config.ProgressOutput, stringid.TruncateID(img.ID), err.Error())
		return err
	}
	return nil
}

func (p *v1Puller) pullImage(ctx context.Context, v1ID, endpoint string, localNameRef reference.Named, layersDownloaded *bool) (err error) {
	var history []string
	history, err = p.session.GetRemoteHistory(v1ID, endpoint)
	if err != nil {
		return err
	}
	if len(history) < 1 {
		return fmt.Errorf("empty history for image %s", v1ID)
	}
	progress.Update(p.config.ProgressOutput, stringid.TruncateID(v1ID), "Pulling dependent layers")

	var (
		descriptors []xfer.DownloadDescriptor
		newHistory  []image.History
		imgJSON     []byte
		imgSize     int64
	)

	// Iterate over layers, in order from bottom-most to top-most. Download
	// config for all layers and create descriptors.
	for i := len(history) - 1; i >= 0; i-- {
		v1LayerID := history[i]
		imgJSON, imgSize, err = p.downloadLayerConfig(v1LayerID, endpoint)
		if err != nil {
			return err
		}

		// Create a new-style config from the legacy configs
		h, err := v1.HistoryFromConfig(imgJSON, false)
		if err != nil {
			return err
		}
		newHistory = append(newHistory, h)

		layerDescriptor := &v1LayerDescriptor{
			v1LayerID:        v1LayerID,
			indexName:        p.repoInfo.Index.Name,
			endpoint:         endpoint,
			v1IDService:      p.v1IDService,
			layersDownloaded: layersDownloaded,
			layerSize:        imgSize,
			session:          p.session,
		}

		descriptors = append(descriptors, layerDescriptor)
	}

	rootFS := image.NewRootFS()
	resultRootFS, release, err := p.config.DownloadManager.Download(ctx, *rootFS, "", descriptors, p.config.ProgressOutput)
	if err != nil {
		return err
	}
	defer release()

	config, err := v1.MakeConfigFromV1Config(imgJSON, &resultRootFS, newHistory)
	if err != nil {
		return err
	}

	imageID, err := p.config.ImageStore.Put(config)
	if err != nil {
		return err
	}

	if p.config.ReferenceStore != nil {
		if err := p.config.ReferenceStore.AddTag(localNameRef, imageID, true); err != nil {
			return err
		}
	}

	return nil
}

func (p *v1Puller) downloadLayerConfig(v1LayerID, endpoint string) (imgJSON []byte, imgSize int64, err error) {
	progress.Update(p.config.ProgressOutput, stringid.TruncateID(v1LayerID), "Pulling metadata")

	retries := 5
	for j := 1; j <= retries; j++ {
		imgJSON, imgSize, err := p.session.GetRemoteImageJSON(v1LayerID, endpoint)
		if err != nil && j == retries {
			progress.Update(p.config.ProgressOutput, stringid.TruncateID(v1LayerID), "Error pulling layer metadata")
			return nil, 0, err
		} else if err != nil {
			time.Sleep(time.Duration(j) * 500 * time.Millisecond)
			continue
		}

		return imgJSON, imgSize, nil
	}

	// not reached
	return nil, 0, nil
}

type v1LayerDescriptor struct {
	v1LayerID        string
	indexName        string
	endpoint         string
	v1IDService      *metadata.V1IDService
	layersDownloaded *bool
	layerSize        int64
	session          *registry.Session
	tmpFile          *os.File
}

func (ld *v1LayerDescriptor) Key() string {
	return "v1:" + ld.v1LayerID
}

func (ld *v1LayerDescriptor) ID() string {
	return stringid.TruncateID(ld.v1LayerID)
}

func (ld *v1LayerDescriptor) DiffID() (layer.DiffID, error) {
	return ld.v1IDService.Get(ld.v1LayerID, ld.indexName)
}

func (ld *v1LayerDescriptor) Download(ctx context.Context, progressOutput progress.Output) (io.ReadCloser, int64, error) {
	progress.Update(progressOutput, ld.ID(), "Pulling fs layer")
	layerReader, err := ld.session.GetRemoteImageLayer(ld.v1LayerID, ld.endpoint, ld.layerSize)
	if err != nil {
		progress.Update(progressOutput, ld.ID(), "Error pulling dependent layers")
		if uerr, ok := err.(*url.Error); ok {
			err = uerr.Err
		}
		if terr, ok := err.(net.Error); ok && terr.Timeout() {
			return nil, 0, err
		}
		return nil, 0, xfer.DoNotRetry{Err: err}
	}
	*ld.layersDownloaded = true

	ld.tmpFile, err = ioutil.TempFile("", "GetImageBlob")
	if err != nil {
		layerReader.Close()
		return nil, 0, err
	}

	reader := progress.NewProgressReader(ioutils.NewCancelReadCloser(ctx, layerReader), progressOutput, ld.layerSize, ld.ID(), "Downloading")
	defer reader.Close()

	_, err = io.Copy(ld.tmpFile, reader)
	if err != nil {
		ld.Close()
		return nil, 0, err
	}

	progress.Update(progressOutput, ld.ID(), "Download complete")

	logrus.Debugf("Downloaded %s to tempfile %s", ld.ID(), ld.tmpFile.Name())

	ld.tmpFile.Seek(0, 0)

	// hand off the temporary file to the download manager, so it will only
	// be closed once
	tmpFile := ld.tmpFile
	ld.tmpFile = nil

	return ioutils.NewReadCloserWrapper(tmpFile, func() error {
		tmpFile.Close()
		err := os.RemoveAll(tmpFile.Name())
		if err != nil {
			logrus.Errorf("Failed to remove temp file: %s", tmpFile.Name())
		}
		return err
	}), ld.layerSize, nil
}

func (ld *v1LayerDescriptor) Close() {
	if ld.tmpFile != nil {
		ld.tmpFile.Close()
		if err := os.RemoveAll(ld.tmpFile.Name()); err != nil {
			logrus.Errorf("Failed to remove temp file: %s", ld.tmpFile.Name())
		}
		ld.tmpFile = nil
	}
}

func (ld *v1LayerDescriptor) Registered(diffID layer.DiffID) {
	// Cache mapping from this layer's DiffID to the blobsum
	ld.v1IDService.Set(ld.v1LayerID, ld.indexName, diffID)
}
